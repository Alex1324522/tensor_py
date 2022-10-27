# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

from copy import deepcopy

from . import uff_pb2 as uff_pb
from .core_descriptor import CORE_DESCRIPTOR, CUSTOM_DESCRIPTOR
from .data import FieldType, create_data, convert_to_debug_data
from .exceptions import UffOrdersException, UffReferenceException
from .graph import Graph


class MetaGraph(object):
    _VERSION = 0x1

    _UFF_STR_ORDER_LIST = ["NC+", "KC+", "NC", "KC"]

    def __init__(self):
        self.referenced_data = {}
        self.graphs = []
        self.descriptor = deepcopy(CORE_DESCRIPTOR)
        self.custom_descriptor_added = False

    def extend_descriptor(self, descriptor):
        self.descriptor.extend_descriptor(descriptor)

    def add_graph(self, name):
        graph = Graph(self, name)
        self.graphs.append(graph)
        return graph

    def to_uff(self, debug=False):
        if debug:
            referenced_data = {k: convert_to_debug_data(v) for k, v in self.referenced_data.items()}
        else:
            referenced_data = self.referenced_data

        return uff_pb.MetaGraph(
            version=self._VERSION,
            descriptor_core_version=self.descriptor.version,
            descriptors=[desc.to_uff(debug) for desc in self.descriptor.descriptors_extended],
            graphs=[graph.to_uff(debug) for graph in self.graphs],
            referenced_data=referenced_data)

    def enable_custom_descriptor(self):
        if self.custom_descriptor_added:
            return

        self.custom_descriptor_added = True
        self.extend_descriptor(CUSTOM_DESCRIPTOR)

    def create_ref(self, prefix, data):
        registered_prefixes = ["orders_"]
        for registered_prefix in registered_prefixes:
            if prefix.startswith(registered_prefix):
                raise UffReferenceException("{} starts with the registered prefix {}"
                                            .format(prefix, registered_prefix))

        if prefix not in self.referenced_data:
            self.referenced_data[prefix] = data
            return uff_pb.Data(ref=prefix)

        idx = 0
        while True:
            key = "%s_%d" % (prefix, idx)
            if key not in self.referenced_data:
                self.referenced_data[key] = data
                return uff_pb.Data(ref=key)
            idx += 1

    def create_orders_ref(self, orders_list, uff_str_extend=None):
        uff_str_dict = {self._create_orders_key(s): s for s in self._UFF_STR_ORDER_LIST}

        if uff_str_extend:
            for uff_str in uff_str_extend:
                key = self._create_orders_key(uff_str)
                if key in uff_str_dict and uff_str_dict[key] != uff_str:
                    raise UffOrdersException("This order {} cannot be added {} already present"
                                             .format(uff_str, uff_str_dict[key]))
                uff_str_dict[key] = uff_str

        ret_orders_list = []
        for orders in orders_list:
            ret_orders_list.append(self._create_orders(orders, uff_str_dict))

        orders_data = create_data(ret_orders_list, FieldType.dim_orders_list)
        orders_ref = "orders_" + "_".join(orders_list)

        if orders_ref in self.referenced_data:
            assert(orders_data == self.referenced_data[orders_ref])
        else:
            self.referenced_data[orders_ref] = orders_data
        return uff_pb.Data(ref=orders_ref)

    @classmethod
    def _create_orders(cls, orders, uff_str_dict):
        orders_key = cls._create_orders_key(orders)
        if orders_key not in uff_str_dict:
            raise UffOrdersException("%s cannot be converted from a known uff string" % orders)

        uff_orders_dict = {}
        spatial_dims_found = False
        uff_str = uff_str_dict[orders_key]
        for idx, c in enumerate(uff_str):
            # UFF main order definition of "+" and "-"
            if c == "+":
                uff_orders_dict["+"] = [idx, uff_pb.OE_INCREMENT]
                uff_orders_dict["-"] = [uff_pb.OE_DECREMENT, idx]
                spatial_dims_found = True
            elif c == "-":
                uff_orders_dict["-"] = [idx, uff_pb.OE_INCREMENT]
                uff_orders_dict["+"] = [uff_pb.OE_DECREMENT, idx]
                spatial_dims_found = True
            elif spatial_dims_found:
                uff_orders_dict[c] = [idx - len(uff_str)]
            else:
                uff_orders_dict[c] = [idx]

        # convert the order based of the UFF main order definition
        orders_list = []
        for c in orders:
            orders_list.extend(uff_orders_dict[c])

        return {uff_pb.OE_SPECIAL: orders_list}

    @staticmethod
    def _create_orders_key(orders):
        orders_set = set(orders)
        if not len(orders_set) == len(orders) or ("-" in orders_set and "+" in orders_set):
            raise UffOrdersException("%s is not a correct order" % orders)

        if "-" in orders_set:
            orders_set.remove("-")
            orders_set.add("+")

        return frozenset(orders_set)

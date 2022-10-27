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

import traceback

from . import uff_pb2 as uff_pb
from .data import create_data, convert_to_debug_data


class Node(object):

    def __init__(self, graph, op, name, inputs=None, fields=None, extra_fields=None):
        self.graph = graph
        self.inputs = inputs if inputs else []
        self.fields = fields if fields else {}
        self.extra_fields = extra_fields if extra_fields else {}
        self.name = name
        self.op = op
        self._trace = traceback.format_stack()[:-1]

    def _convert_fields(self, fields, debug):
        descriptor = self.graph.meta_graph.descriptor

        ret_fields = {}
        for k, v in fields.items():
            if v is None:
                continue
            if not isinstance(v, uff_pb.Data):
                if self.op in descriptor:
                    field_type = descriptor[self.op].get_field_type(k)
                    ret_fields[k] = create_data(v, field_type)
                else:
                    ret_fields[k] = create_data(v)
            else:
                ret_fields[k] = v
            if debug:
                ret_fields[k] = convert_to_debug_data(ret_fields[k])
        return ret_fields

    def to_uff(self, debug=False):
        return uff_pb.Node(id=self.name,
                           inputs=[i.name if isinstance(i, Node) else i for i in self.inputs],
                           operation=self.op,
                           fields=self._convert_fields(self.fields, debug),
                           extra_fields=self._convert_fields(self.extra_fields, debug))

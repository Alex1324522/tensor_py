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

import numpy as np

from . import uff_pb2 as uff_pb
from .utils import SimpleObject, int_types
from .exceptions import UffException

FieldType = SimpleObject({field.name: field.name for field in uff_pb.Data.DESCRIPTOR.fields})


class List(list):

    def __init__(self, list_type, *args, **kwargs):
        self.list_type = list_type
        return super(List, self).__init__(*args, **kwargs)


def infer_field_type(elt):
    if isinstance(elt, str):
        return FieldType.s
    if isinstance(elt, bool):
        return FieldType.b
    if isinstance(elt, int_types):
        return FieldType.i
    if isinstance(elt, float):
        return FieldType.d
    if isinstance(elt, List):
        if type(elt.list_type) is type:
            return str(infer_field_type(elt.list_type())) + "_list"
        if not elt.list_type.endswith("_list"):
            return str(elt.list_type) + "_list"
    if isinstance(elt, list):
        raise UffException("unsupported list type")

    if isinstance(elt, np.dtype):
        elt = elt.type
    if isinstance(elt, type) and issubclass(elt, np.number):
        return FieldType.dtype

    return ""


_DTYPE_NP_TO_UFF = {
    np.int8: uff_pb.DT_INT8,
    np.int16: uff_pb.DT_INT16,
    np.int32: uff_pb.DT_INT32,
    np.int64: uff_pb.DT_INT64,
    np.float16: uff_pb.DT_FLOAT16,
    np.float32: uff_pb.DT_FLOAT32
}


def _create_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype in _DTYPE_NP_TO_UFF:
        return _DTYPE_NP_TO_UFF[dtype]
    if dtype not in uff_pb.DataType.values() or dtype == uff_pb.DT_INVALID:
        raise UffException("dtype {} unknown".format(dtype))
    return dtype


def _create_dim_orders(orders):
    for k, order in orders.items():
        if not isinstance(order, uff_pb.ListINT64):
            orders[k] = uff_pb.ListINT64(val=order)
    return uff_pb.DimensionOrders(orders=orders)


def _create_list_dim_orders(val):
    dims_orders_list = []
    for dim_orders in val:
        if isinstance(dim_orders, uff_pb.DimensionOrders):
            dims_orders_list.append(dim_orders)
        else:
            dims_orders_list.append(_create_dim_orders(dim_orders))
    return uff_pb.ListDimensionOrders(val=dims_orders_list)


_CTOR_LIST = {
    FieldType.s_list: uff_pb.ListString,
    FieldType.b_list: uff_pb.ListBool,
    FieldType.d_list: uff_pb.ListDouble,
    FieldType.i_list: uff_pb.ListINT64,
    FieldType.dtype_list: uff_pb.ListDataType,
    FieldType.dim_orders_list: _create_list_dim_orders,
}


def create_data(elt, field_type=None):
    if elt is None:
        return uff_pb.Data()

    if field_type is None:
        field_type = infer_field_type(elt)

        # FIXME: All of this
    assert(field_type != FieldType.ref)
    # if __debug__:
    #     print("Creating data of type: " + str(type(elt)) + " given FieldType: " + field_type)

    if field_type.endswith("_list"):
        try:
            return uff_pb.Data(**{field_type: _CTOR_LIST[field_type](val=elt)})
        except Exception:
            return uff_pb.Data()

    if field_type == FieldType.dim_orders:
        return uff_pb.Data(dim_orders=_create_dim_orders(elt))

    if field_type == FieldType.dtype:
        try:
            return uff_pb.Data(dtype=_create_dtype(elt))
        except Exception:
            return uff_pb.Data(dtype=7)
    try:
        return uff_pb.Data(**{field_type: elt})
    except Exception:
        return uff_pb.Data()


def convert_to_debug_data(data):
    if data.WhichOneof("data_oneof") == FieldType.blob and len(data.blob) > 32:
        return uff_pb.Data(blob=str.encode("(...%d bytes skipped...)" % len(data.blob)))
    return data

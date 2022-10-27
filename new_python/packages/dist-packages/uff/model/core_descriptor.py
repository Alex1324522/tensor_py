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

from . import uff_pb2 as uff_pb
from .descriptor import Descriptor, DescriptorOp
from .data import FieldType


CORE_DESCRIPTOR = Descriptor(None, 0x1, False, {
    "Input":
        DescriptorOp().inputs_size(0)
        .field(FieldType.dtype, "dtype", uff_pb.DT_FLOAT32)
        .field(FieldType.i_list, "shape"),

    "Identity":
        DescriptorOp().inputs_size(1),

    "Const":
        DescriptorOp().inputs_size(0)
        .field(FieldType.blob, "values").ref_field("values")  # force reference checking
        .field(FieldType.dtype, "dtype")
        .field(FieldType.i_list, "shape"),

    "Conv":
        DescriptorOp().inputs_size(2)  # input, weights
        .fieldOrders()
        .field(FieldType.i, "number_groups", 1)
        .field(FieldType.i_list, "dilation", [])  # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "ConvTranspose":
        DescriptorOp().inputs_size(3)  # input, weights, shape
        .fieldOrders(2)
        .field(FieldType.i, "number_groups", 1)
        .field(FieldType.i_list, "dilation", [])  # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "Pool":
        DescriptorOp().inputs_size(1)
        .fieldOrders()
        .field_enum("func", ["max", "avg"])
        .field(FieldType.i_list, "kernel", [])    # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "FullyConnected":
        DescriptorOp().inputs_size(2)  # input, weights
        .fieldOrders(),

    "LRN":
        DescriptorOp().inputs_size(1)
        .fieldOrders()
        .field(FieldType.i, "window_size")
        .field(FieldType.d, "alpha")
        .field(FieldType.d, "beta")
        .field(FieldType.d, "k"),

    "Binary":
        DescriptorOp().inputs_size(2)
        .field_enum("func", ["min", "max", "mul", "sub", "div", "add", "pow"]),

    "Unary":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["neg", "exp", "log", "abs", "sqrt",
            "rsqrt", "square", "sin", "cos", "tan", "sinh", "cosh",
            "asin", "acos", "atan", "asinh", "acosh", "atanh", "ceil", "floor"]),

    "ExpandDims":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "ArgMax":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "ArgMin":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "Reshape":
        DescriptorOp().inputs_size(2),  # input, shape

    "Transpose":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i_list, "permutation"),

    "Reduce":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["sum", "prod", "max", "min", "mean"])
        .field(FieldType.i_list, "axes")
        .field(FieldType.b, "keepdims"),

    "Concat":
        DescriptorOp().has_inputs()
        .field(FieldType.i, "axis"),

    "Stack":
        DescriptorOp().has_inputs()
        .field(FieldType.i, "axis"),

    "Shape":
        DescriptorOp().inputs_size(1),

    "StridedSlice":
        DescriptorOp().inputs_size(4)   # input, begin, end, strides
        .field(FieldType.i, "begin_mask")
        .field(FieldType.i, "end_mask")
        .field(FieldType.i, "shrink_axis_mask"),

    "MarkOutput":
        DescriptorOp().has_inputs(),

    # TODO
    # LCN
    # Select
    # Embed

    # TODO: Temporary - to remove when graph pattern match will be implemented in TensoRT importer
    # we will keep the helper function in the Graph for those though
    "Activation":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["relu", "relu6", "sigmoid", "tanh", "elu", "selu", "softsign", "softplus"]),

    # Required to maintain backwards compatibility
    "LeakyRelu":
        DescriptorOp().inputs_size(1)
        .field(FieldType.d, "alpha"),

    "Softmax":
        DescriptorOp().inputs_size(1)
        .fieldOrders(1)
        .field(FieldType.i, "axis"),

    "BatchNorm":
        DescriptorOp().inputs_size(5)  # input, gamma, beta, moving_mean, moving_variance
        .fieldOrders(1)
        .field(FieldType.d, "epsilon"),

    "Squeeze":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i_list, "axes", []),

    "Flatten":
        DescriptorOp().inputs_size(1),

    "Pad":
        DescriptorOp().inputs_size(2),  # input, padding

    "Gather":
        DescriptorOp().inputs_size(2)
        .field(FieldType.dtype, "indices_dtype")
        .field(FieldType.dtype, "params_dtype")
        .field(FieldType.b, "validate_indices"),  # indices, input

    "GatherV2":
        DescriptorOp().inputs_size(2)
        .field(FieldType.i, "axis")
        .field(FieldType.dtype, "indices_dtype")
        .field(FieldType.dtype, "params_dtype"),
    # END TODO
})


CUSTOM_DESCRIPTOR = Descriptor("custom", 0x1, False, {}).add_regex_operator("_.+")

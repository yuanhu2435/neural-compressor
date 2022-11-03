#
#  -*- coding: utf-8 -*-
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import json
import pdb
import numpy as np
import time
import argparse
import yaml
import shutil
import unittest
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from neural_compressor.utils.utility import LazyImport, CpuInfo
from packaging.version import Version
from neural_compressor.experimental import MixedPrecision
from neural_compressor import conf

print(tf.__version__)

batchForRandomInput = {
"placeholder_23": 1,
"placeholder_7": 1,
"placeholder_49": 1,
"placeholder_59": 1,
"placeholder_9": 1,
"placeholder_92": 1,
"placeholder_82": 1,
"placeholder_60": 1,
"placeholder_67": 1,
"placeholder_38": 1,
"placeholder_61": 1,
"placeholder_11": 1,
"placeholder_76": 1,
"placeholder_24": 1,
"placeholder_46": 1,
"placeholder_68": 1,
"placeholder_81": 1,
"placeholder_62": 1,
"placeholder_90": 1,
"placeholder_13": 1,
"placeholder_42": 1,
"placeholder_14": 1,
"placeholder_85": 1,
"placeholder_78": 1,
"placeholder_57": 1,
"placeholder_55": 1,
"placeholder_98": 1,
"placeholder_56": 1,
"placeholder_86": 1,
"placeholder_35": 1,
"placeholder_26": 1,
"placeholder_44": 1,
"placeholder_97": 1,
"placeholder_95": 1,
"placeholder_21": 1,
"placeholder_63": 1,
"placeholder_54": 1,
"placeholder_65": 1,
"placeholder_84": 1,
"placeholder_66": 1,
"placeholder_29": 1,
"placeholder_91": 1,
"placeholder_40": 1,
"placeholder_50": 1,
"placeholder_2": 1,
"placeholder_64": 1,
"placeholder_37": 1,
"placeholder_8": 1,
"placeholder_41": 1,
"placeholder_43": 1,
"placeholder_80": 1,
"placeholder_73": 1,
"placeholder_77": 1,
"placeholder_17": 1,
"placeholder_3": 1,
"placeholder_39": 1,
"placeholder_19": 1,
"placeholder_45": 1,
"placeholder_1": 1,
"placeholder_18": 1,
"placeholder_30": 1,
"placeholder_27": 1,
"placeholder_28": 1,
"placeholder_94": 1,
"placeholder_12": 1,
"placeholder_74": 1,
"placeholder_75": 1,
"placeholder_47": 1,
"placeholder_51": 1,
"placeholder_36": 1,
"placeholder_33": 1,
"placeholder_20": 1,
"placeholder_5": 1,
"placeholder_48": 1,
"placeholder_15": 1,
"placeholder_16": 1,
"placeholder_4": 1,
"placeholder_70": 1,
"placeholder_71": 1,
"placeholder_69": 1,
"placeholder_31": 1,
"placeholder_32": 1,
"placeholder_72": 1,
"placeholder_52": 1,
"placeholder_96": 1,
"placeholder_93": 1,
"placeholder_58": 1,
"placeholder_22": 1,
"placeholder_87": 1,
"placeholder_88": 1,
"placeholder_89": 1,
"placeholder_83": 1,
"placeholder_6": 1,
"placeholder_10": 1,
"placeholder_25": 1,
"placeholder_34": 1,
"placeholder_53": 1,
"placeholder_79": 1,
}

def get_input_nodes(graph_def):
    nodes = []
    for node in graph_def.node:
        nodename = node.name
        if node.op == 'Placeholder' :
            nodes.append(node)

    return nodes

def build_tf_graph():
    input_node = node_def_pb2.NodeDef()
    input_node.name = "input"
    input_node.op = "Placeholder"
    input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))

    conv1_weight_node = node_def_pb2.NodeDef()
    conv1_weight_node.name = "conv1_weights"
    conv1_weight_node.op = "Const"
    conv1_weight_value = np.float32(np.abs(np.random.randn(3,3,3,32)))
    conv1_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv1_weight_value, conv1_weight_value.dtype.type, conv1_weight_value.shape)))

    conv1_node = node_def_pb2.NodeDef()
    conv1_node.name = "conv1"
    conv1_node.op = "Conv2D"
    conv1_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv1_node.input.extend([input_node.name, conv1_weight_node.name])
    conv1_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv1_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv1_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv1_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    bias_node = node_def_pb2.NodeDef()
    bias_node.name = "conv1_bias"
    bias_node.op = "Const"
    bias_value = np.float32(np.abs(np.random.randn(32)))
    bias_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        bias_value, bias_value.dtype.type, bias_value.shape)))

    bias_add_node = node_def_pb2.NodeDef()
    bias_add_node.name = "conv1_bias_add"
    bias_add_node.op = "BiasAdd"
    bias_add_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])
    bias_add_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node.input.extend([bias_add_node.name])

    conv2_weight_node = node_def_pb2.NodeDef()
    conv2_weight_node.name = "conv2_weights"
    conv2_weight_node.op = "Const"
    conv2_weight_value = np.float32(np.abs(np.random.randn(3,3,32,32)))
    conv2_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv2_weight_value, conv2_weight_value.dtype.type, conv2_weight_value.shape)))

    conv2_node = node_def_pb2.NodeDef()
    conv2_node.name = "conv2"
    conv2_node.op = "Conv2D"
    conv2_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv2_node.input.extend([relu_node.name, conv2_weight_node.name])
    conv2_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv2_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv2_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv2_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    bias_node2 = node_def_pb2.NodeDef()
    bias_node2.name = "conv2_bias"
    bias_node2.op = "Const"
    bias_value2 = np.float32(np.abs(np.random.randn(32)))
    bias_node2.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node2.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
        bias_value2, bias_value2.dtype.type, bias_value2.shape)))

    bias_add_node2 = node_def_pb2.NodeDef()
    bias_add_node2.name = "conv2_bias_add"
    bias_add_node2.op = "BiasAdd"
    bias_add_node2.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node2.input.extend([conv2_node.name, bias_node2.name])
    bias_add_node2.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    relu_node2 = node_def_pb2.NodeDef()
    relu_node2.op = "Relu"
    relu_node2.name = "relu2"
    relu_node2.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node2.input.extend([bias_add_node2.name])

    conv3_weight_node = node_def_pb2.NodeDef()
    conv3_weight_node.name = "conv3_weights"
    conv3_weight_node.op = "Const"
    conv3_weight_value = np.float32(np.abs(np.random.randn(3,3,32,32)))
    conv3_weight_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            conv3_weight_value, conv3_weight_value.dtype.type, conv3_weight_value.shape)))

    conv3_node = node_def_pb2.NodeDef()
    conv3_node.name = "conv3"
    conv3_node.op = "Conv2D"
    conv3_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    conv3_node.input.extend([relu_node2.name, conv3_weight_node.name])
    conv3_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv3_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
    conv3_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
    conv3_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))

    identity_node = node_def_pb2.NodeDef()
    identity_node.name = "final"
    identity_node.op = "Identity"
    identity_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    identity_node.input.extend([conv3_node.name])

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend([input_node,
                            conv1_weight_node,
                            conv1_node,
                            bias_node,
                            bias_add_node,
                            #cast_node,
                            relu_node,
                            #cast2_node,
                            conv2_weight_node,
                            conv2_node,
                            bias_node2,
                            bias_add_node2,
                            relu_node2,
                            conv3_weight_node,
                            conv3_node,
                            identity_node
                            ])
    return test_graph

class TestMixedPrecisionOnNonEnabledHost(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print(CpuInfo().bf16)

    @unittest.skipIf(CpuInfo().bf16, 'skip since hardware support bf16')
    def test_on_non_enabled_host_tf(self):
        from neural_compressor.experimental import MixedPrecision
        from neural_compressor import conf
        conf.model.framework = 'tensorflow'
        converter = MixedPrecision(conf)
        converter.model = 'zhenyun.pb'
        converter.precisions = 'bf16'
        output_model = converter.fit()
        output_model.save('zhenyun_output_model')
    
if __name__ == "__main__":
    unittest.main()


import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = """
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: mse
            accuracy_criterion:
              relative: 0.01
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestFoldPadConv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove("fake_yaml.yaml")

    @disable_random()
    def test_fold_pad_conv(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(x, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization("fake_yaml.yaml")
            dataset = quantizer.dataset("dummy", shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            found_pad = False

            for i in output_graph.graph_def.node:
                if i.op == "Pad":
                    found_pad = True
                    break
            self.assertEqual(found_pad, False)

    @disable_random()
    def test_fold_non_const_pad_conv(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        vec = tf.raw_ops.DataFormatVecPermute(x=paddings, src_format="NHWC", dst_format="NHWC")
        x_pad = tf.pad(x, vec, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 16, 16], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        relu = tf.nn.relu(normed, name="op_to_store")
        out_name = relu.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization("fake_yaml.yaml")
            dataset = quantizer.dataset("dummy", shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            found_pad = False

            for i in output_graph.graph_def.node:
                if i.op == "Pad":
                    found_pad = True
                    break
            self.assertEqual(found_pad, False)


if __name__ == "__main__":
    unittest.main()

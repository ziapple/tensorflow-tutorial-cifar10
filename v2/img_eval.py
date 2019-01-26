# -*- coding: utf-8 -*-

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by img_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import v2.img_network as img_network

FLAGS = tf.app.flags.FLAGS

# 验证图片过程事件保存路径
tf.app.flags.DEFINE_string('eval_dir', './img_eval', """Directory where to write event logs.""")
# 为test，表示验证图片，读取./img_input下面的图片，否则读取./cifar_train下面的训练图片，开始训练
tf.app.flags.DEFINE_string('eval_data', 'test1', """Either 'test' or 'train_eval'.""")
# 训练网络保存的路径
tf.app.flags.DEFINE_string('checkpoint_dir', '../cifar10_train', """Directory where to read model checkpoints.""")
# 验证图片频率，10s读取一次
tf.app.flags.DEFINE_integer('eval_interval_secs', 10, """How often to run the eval.""")
# 验证样本图片总的数量，必须大于batch_size
tf.app.flags.DEFINE_integer('num_examples', 10, """Number of examples to run.""")
# 是否只验证一次，如果为false，配合eval_interval_secs使用
tf.app.flags.DEFINE_boolean('run_once', True, """Whether to run eval only once.""")


# 验证一次方法
# Run Eval once.
#  Args:
#    saver: Saver.
#    summary_writer: Summary writer.
#    top_k_op: Top K op.
#    summary_op: Summary op.
def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        # 还原训练参数
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # 每批次验证的数量
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / FLAGS.num_examples
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
          coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


# 评估图片
# Eval CIFAR-10 for a number of steps.
def evaluate():
    with tf.Graph().as_default():
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = img_network.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = img_network.inference(images)

        # Calculate predictions.
        # in_top_k(predictions, targets, k, name=None)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(img_network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        # specifies the variables that will be saved and restored
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph_def=graph_def)

        while True:
            print("start evaluate the precision with examples %s" % FLAGS.num_examples)
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


# 对image-test下的模型进行评估
def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

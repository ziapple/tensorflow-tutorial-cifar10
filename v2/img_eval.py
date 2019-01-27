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

"""
训练了大概一万张图片，用cifar的图片准确率80%左右
网上自己找的图片，准确率只有50%左右,对飞机、船、汽车这些区别度比较大的识别较好，对马、狐狸、狗、猫这些背景比较复杂的识别度不太好
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
import scipy.misc

FLAGS = tf.app.flags.FLAGS

# 验证图片过程事件保存路径
tf.app.flags.DEFINE_string('eval_dir', './img_eval', """Directory where to write event logs.""")
# 为test，表示验证图片，读取./img_input下面的图片，否则读取./cifar_train下面的训练图片，开始训练
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
# 训练网络保存的路径
tf.app.flags.DEFINE_string('checkpoint_dir', '../cifar10_train', """Directory where to read model checkpoints.""")
# 验证图片频率，10s读取一次
tf.app.flags.DEFINE_integer('eval_interval_secs', 10, """How often to run the eval.""")
# 验证样本图片总的数量，必须大于batch_size
tf.app.flags.DEFINE_integer('num_examples', 13, """Number of examples to run.""")
# 是否只验证一次，如果为false，配合eval_interval_secs使用
tf.app.flags.DEFINE_boolean('run_once', True, """Whether to run eval only once.""")

# 读取标签文件
def load_labels(filename):
    with open(filename, 'rb') as f:
        lines = [x.replace("\n", "") if x != "\n" else x for x in f.readlines()]
        return lines


# 验证一次方法
# Run Eval once.
#  Args:
#    saver: Saver.
#    summary_writer: Summary writer.
#    top_k_op: Top K op.
#    summary_op: Summary op.
def eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, images):
    with tf.Session() as sess:
        # 还原训练参数
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # 启动 Coordinator 等待线程 runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # 迭代次数=总测试样本数/每批次样本数
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            # 计算测试命中数量
            true_count = 0
            step = 0
            lines = load_labels("../cifar10_data/cifar-10-batches-bin/batches.meta.txt")
            while step < num_iter and not coord.should_stop():
                # 调用sess.run()会触发计算一个批次样本，再调用的时候会取下一个批次，top_k_op看不到预测中间过程
                # 为了看到模型的预测结果，这里用logits来run
                # predictions = sess.run([top_k_op])
                softmaxs, results, _images = sess.run([logits, labels, images])
                i = 0
                for result in results:
                    print('start %d,%s=%s(img_eval/tmp/%d.%s-%s.jpg)' % (step*FLAGS.batch_size + i,
                                                                  lines[np.argmax(softmaxs[i])], lines[result],
                                                                         step * FLAGS.batch_size + i, lines[result], result))
                    #scipy.misc.toimage(_images[i]).save('img_eval/tmp/%d.%s-%s.jpg' % (step*FLAGS.batch_size + i, lines[result], result))
                    prediction = np.equal(np.argmax(softmaxs[i]), result)
                    true_count += np.sum(prediction)
                    # print(softmaxs[i], result)
                    i += 1
                #true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / FLAGS.num_examples
            print('%s: total accuracy of prediction = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
          coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


# 测试验证图片
def evaluate():
    with tf.Graph().as_default():
        # 获取测试集还是训练集作为验证样本,'test'表示测试集
        eval_data = FLAGS.eval_data == 'test'
        images, labels = img_network.inputs(eval_data=eval_data)

        # 获取训练模型，用来预测
        logits = img_network.inference(images)

        # 作用是返回一个布尔向量，说明目标值是否存在于预测值之中，
        # top_k表示logits预测的最大值所在的位置，和label做比对，k查找最大的K个值
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # 存储学习变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(img_network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        # 定义存储器
        saver = tf.train.Saver(variables_to_restore)

        # 构建统计量
        summary_op = tf.merge_all_summaries()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph_def=graph_def)

        while True:
            print("start evaluate the precision with examples %s" % FLAGS.num_examples)
            eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, images)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


# 对模型进行评估
def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

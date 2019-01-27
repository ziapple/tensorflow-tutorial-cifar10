# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import img_input
import scipy.misc

filename = os.path.join('../cifar10_data', "img_eval.bin")


# 读取标签文件
def load_labels(filename):
    with open(filename, 'rb') as f:
        lines = [x.replace("\n", "") if x != "\n" else x for x in f.readlines()]
        print(lines)
        return lines


# 简单的测试input格式是否正确
def test_simple():
    lines = load_labels("../cifar10_data/cifar-10-batches-bin/batches.meta.txt")
    with tf.Session() as sess:
      q = tf.FIFOQueue(99, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      result = img_input.read_cifar10(q)
      key, label, uint8image = sess.run([result.key, result.label, result.uint8image])
      print('../cifar10_data/raw/%s-%d.jpg' % (lines[label[0]], label[0]))
      scipy.misc.toimage(uint8image).save('../cifar10_data/raw/%s-%d.jpg' % (lines[label[0]], label[0]))


if __name__ == "__main__":
    test_simple()
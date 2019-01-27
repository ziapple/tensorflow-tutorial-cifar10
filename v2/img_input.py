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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

# 用于训练的真实图像大小，如果这个值改变，整个模型需要重新训练
IMAGE_SIZE = 24
# 图像分类的大小，跟图像标签大小一致，用于softmax分类
NUM_CLASSES = 10
# 训练的样本总数量，5万张
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# 验证的样本总数量，1万张
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# 每批次训练的图片数量,该数字必须小于训练和测试集总的数量，否则测试的时候准确率会>1,
tf.app.flags.DEFINE_integer('batch_size', 1, """Number of images to process in a batch.""")
# 训练图片所在目录,跟训练集在同一个目录，这样验证的时候好用，既可以用训练集来验证，也可以用测试集来验证
tf.app.flags.DEFINE_string('data_dir', '../cifar10_data', """Path to the images data directory.""")


# 读取cifar10的图片数据.
# 建议: 如果同时N个线程读取数据，可以调用N次，该方法会由N个Reader读取不同文件的不同位置，会得到更好的混合样本
# 参数
#  filename_queue: 要读取的文件名称队列.
# 返回
#  代表（一个！！！）样本的对象
#    height: 结果中的行数Rows
#    width: 结果中的列数Columns
#    depth: 图片通道（红、蓝、绿）
#    key: 描述文件和位置的标量，比如data_bin:1表示读取的是data_bin文件的第一张图片
#    label: 0-9范围的图片标签，类型为int32
#    uint8image: [height, width, depth]的图片数据标量，每个元素类型为uint8
def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # 图片标签占的字节大小，此处为1个字节
    label_bytes = 1  # 2 for CIFAR-100
    # 图片高度
    result.height = 32
    # 图片宽度
    result.width = 32
    # 图片通道数
    result.depth = 3
    # 图片本身占的字节大小，为32*32*3=3072
    image_bytes = result.height * result.width * result.depth
    # 每条记录固定大小，为3073个字节
    record_bytes = label_bytes + image_bytes

    # 读取一条记录，从filename_queue读取文件名filenames，在cifar10格式中
    # 没有header和footer，所以header_bytes和footer_bytes默认为0
    # FixedLengthRecordReader调用时会接着上次读取的位置继续读取文件，而不会从头开始读取。
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # 将value字符串转化为一个uint8的向量，长度为record_bytes的长度3073，每一个byte为0-255
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 第一个字节为标签label，从unit8转化为int32.
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下1-3073字节数据代表image数据, 做变换处理reshape，从[depth * height * width] 转化为 [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # 从 [depth, height, width] 转置为 [height, width, depth].
    # 注意reshape和transcope的区别，一个是向量形状变换（不改变数据），transcope会根据新的纬度转置数据
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


# 构建样本批次队列，把样本填充到队列里面去，训练的时候从这个队列里面去读取
# 参数
#    image: [height, width, 3] 的3D向量，元素类型为type.float32.
#    label: 1D向量，类型为type.int32
#    min_queue_examples: int32, 队列里面最小的样本数量
#    batch_size: 每个批次的样本图片数量，批次出队后，队列继续从样本库里面补充
#    shuffle: 为true通过随机打乱张量的顺序创建批次.
#  返回
#    images: 4D的向量,[batch_size, height, width, 3]
#    labels: 1D向量,[label,label, ...]
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    # 预处理线程数
    num_preprocess_threads = 16
    # 生成队列，shuffles样本，从队列样本里面读取batch_size大小
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size,
                                                     num_threads=num_preprocess_threads,
                                                     capacity=min_queue_examples + 3 * batch_size,
                                                     min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3 * batch_size)

    # 可视化图片统计量，批次训练的图片放到统计量里面去，tensorboard可以实时看到正在训练的图片
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# 使用Reader读取训练样本，并对输入样本进行随机截取、翻转等处理，加强训练样本的多样性
# 该函数每次只负责读取处理一张图片，返回tf.train.batch批处理队列即可，tensorflow会自动批处理
# 参数
#  data_dir: 样本数据目录路径
#  batch_size: 每批次处理的样本数量大小

# 返回
#  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#  labels: Labels. 1D tensor of [batch_size] size.
def distorted_inputs(data_dir, batch_size):
    # 文件名，从data_batch_(1,2,3).bin里面读取
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(3, 6)]
    # 判断文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)

    # 生成tf读取的文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件名队列中读取样本数据，每次读取一个样本
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 图像处理，做了很多图像预处理
    # 随机选取了图像中的一部分，大小为[24,24]
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # 随机水平翻转图片
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 随机调整亮度
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # 随机调整图像对比度
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # 图像白化操作，数字均值变为0，方差变为1
    float_image = tf.image.per_image_whitening(distorted_image)

    # 确保随机shuffle抽取有很好的混合性.
    min_fraction_of_examples_in_queue = 0.4
    # 队列里面最小的样本数，低于这个样本自动补充
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # 填充tensorflow的样本读取队列(images和labels）
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


# 用Reader读取测试验证图片，测试样本（非训练样本）对图像只做截取crop和白化处理，不做翻转、亮度、对比度处理
# 参数
#  eval_data: bool, true表示使用测试样本，false表示使用训练样本来验证
#  data_dir: 测试样本数据目录
#  batch_size: 批次处理的样本数量
# 返回
#  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#  labels: Labels. 1D tensor of [batch_size] size.
def inputs(eval_data, data_dir, batch_size):
    # false则读取训练样本
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:  # 读取测试样本
        filenames = [os.path.join(data_dir, 'img_eval.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # 文件不存在则报错
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 生成读取文件队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件队列里面读取样本，每次读取一个
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 截取图像，大小为[24,24]
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

    # 白化处理图像，样本均值为0，方差为1
    float_image = tf.image.per_image_whitening(resized_image)

    # 确保样本读取队列具有很好的随机性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # 构建样本读取队列，注意，测试样本的时候是按顺序读取
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)

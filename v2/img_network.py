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

"""构建cifar10网络模型
 重要的函数介绍：

 # distored_input()()读取训练数据集，如果是读取验证集,使用input()
 inputs, labels = distorted_inputs()

 # 构建预测模型
 predictions = inference(inputs)

 # 创建整个训练网络的损失函数
 loss = loss(predictions, labels)

 # 创建 graph，开始训练网络
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import v2.img_input as img_input

# 全局变量
FLAGS = tf.app.flags.FLAGS
# 样本图像的大小
IMAGE_SIZE = img_input.IMAGE_SIZE
# 图像分类的大小，跟图像标签大小一致，用于softmax分类
NUM_CLASSES = img_input.NUM_CLASSES
# 训练集总的数量，5万张
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = img_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# 测试集总的数量，1万张
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = img_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 训练模型常量
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# 批量训练大小batch_size必须大于衰减步长，最好是NUM_EPOCHS_PER_DECAY的倍数，否则样本全部训练完了也不会更新
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# 学习率影响因子，学习率=当前学习率*影响因子，每次以0.1比例衰减
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# 初始化学习率
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# 如果模型使用多GPUs训练的, 用tower_0前缀来区别op操作，但在tensorboard的summaries统计量里面是去掉前缀的
TOWER_NAME = 'tower'


# 内部函数，辅助生成活动的统计量summaries
# 生成变量的histogram柱状图,生成变量的sparsity稀疏度
# 参数
#    x: Tensor变量
# 返回
#    无
def _activation_summary(x):
    # 去除'tower_[0-9]/'的变量前缀，避免多GPU训练的时候给变量加了前缀
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# 内部函数，创建在CPU内存上的变量
# 参数
#  name: tensor变量名
#  shape：tensor变量形状
#  initializer：变量的初始化器，可以是正态分布器、随机分布器
# 返回
#  Variable Tensor
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


# 内部函数，weight权重衰退变量初始化，weight在初始化的时候是正态分布的
# 更重要的是，给损失函数加了weight的L2正则化，防止过拟合
# Args:
#    name: tensor变量名
#    shape: tensor变量的形状
#    stddev: 标准差
#    wd: 一个wd矩阵，用来和L2Loss规则化矩阵相乘，为空不相乘
#
#  Returns:
#    Variable Tensor
def _variable_with_weight_decay(name, shape, stddev, wd):
    # 从截断的正态分布中输出随机值,按照mean,std分布随机输出值
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# 读取数据（进行转换处理,读取cifar10训练集
# 返回
#    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#    labels: Labels. 1D tensor of [batch_size] size.
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    return img_input.distorted_inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)


# 读取数据,读取cifar10测试集，不会进行白化、水平翻转、对比度等处理
# 参数
#   eval_data:true，表示测试集，false表示把训练集当成测试集来验证
# 返回
#    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#    labels: Labels. 1D tensor of [batch_size] size.
def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    if not eval_data:
        data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return img_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)


# 构建卷积网络模型
#  Args:
#    images: 从distorted_inputs() or inputs(). 读取的样本集合，包括image本身和label
#
#  Returns:
#    Logits.
# 为了在多个GPU上共享变量，通过tf.get_variable()实例化所有变量，而不是用tf.Variable()
# 如果在单个CPU上跑，可以直接用tf.Variable()
def inference(images):
    # conv1，卷积网络第一层
    with tf.variable_scope('conv1') as scope:
        # 构建一个5*5*3(输入通道)*64(输出通道)的核函数，用正态分布来初始化，wd为0表示不用L2正则化来弥补损失函数
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        # 构建卷积函数，步长为1
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # 构建b偏差量，64个通道输出，和weight的输出通道必须一致
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        # 给卷积函数增加偏差量
        bias = tf.nn.bias_add(conv, biases)
        # 采用relu激活函数
        conv1 = tf.nn.relu(bias, name=scope.name)
        # 生成conv1的统计量
        _activation_summary(conv1)

    # 池化pool1，降低图片纬度，步长为3，池化完后图像变为24/2=12
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # norm1,LRN函数类似DROPOUT和数据增强作为relu激励之后防止数据过拟合而提出的一种处理方法,局部响应标准化
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2,卷积网络第二层
    with tf.variable_scope('conv2') as scope:
        # 构建一个5*5*64(输入通道)*64(输出通道)的核函数，用正态分布来初始化，wd为0表示不用L2正则化来弥补损失函数
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        # 构建卷积函数，步长为1
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        # 构建b偏差量，64个通道输出，和weight的输出通道必须一致
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        # 给卷积函数增加偏差量
        bias = tf.nn.bias_add(conv, biases)
        # 采用relu激活函数
        conv2 = tf.nn.relu(bias, name=scope.name)
        # 生成conv2的统计量
        _activation_summary(conv2)

    # norm2，跟第一层不同，先做norm，再做pool
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2，池化后的图像大小为12/2=6
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3，第三层，构建全连接网络，训练参数weight降维成[dim, 384]
    with tf.variable_scope('local3') as scope:
        # 将所有纬度都铺平，变成FLAGS.batch_size一个纬度，第二个纬度就是images
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        # 获取第二个纬度向量的纬度dim,应该是64个神经元
        dim = reshape.get_shape()[1].value
        # 构建权重矩阵，大小为[dim,384]，注意全连接加了L2正则化损失函数，防止过拟合
        # 注意为什么选取384个参数，第三层有64个神经元，此时图片大小为6*6*3，选取第一个纬度64*6=384个变量
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        # 构建偏移量
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))

    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

    # local4，第四层，一个全连接层，把最后训练参数降为成一维，为后面softmax做好准备
    with tf.variable_scope('local4') as scope:
        # 设置了384/2=192个训练参数，注意用到了L2的Loss函数
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))

    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

    # softmax, i.e. softmax(WX + b)，softmax层，输出分类
    with tf.variable_scope('softmax_linear') as scope:
        # NUM_CLASSES就是图像的分类
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

    # 输出为softmax概率矩阵,大小为样本数*标签数
    return softmax_linear


# 损失函数
#  增加统计量 "Loss" and "Loss/avg".
#  Args:
#    logits: inference()的神经网络输出层的输出，此处是一个softmax函数
#    labels: 样本标签集，大小为[batch_size]
#
#  Returns:
#    Loss向量，float
#  计算某一批次的平均损失值
def loss(logits, labels):
    # 转换为int64
    labels = tf.cast(labels, tf.int64)
    # 计算logits 和 labels 之间的稀疏softmax 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    # 计算各个纬度上的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 给损失函数增加统计量
#  Generates moving average for all losses and associated summaries for
#  visualizing the performance of the network.
#
#  Args:
#    total_loss: Total loss from loss().
#  Returns:
#    loss_averages_op: op for generating moving averages of losses.
#  """
#  # Compute the moving average of all individual losses and the total loss.
def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + '(raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


# 训练网络模型
# 在神经网络的训练过程中，学习率(learning rate)控制着参数的更新速度，
# 越到后面学习率越小，weight更新幅度越小，越精确
# Train CIFAR-10 model.
#  生成参数优化器，适用于所有训练参数，增加移动平均数
#  Args:
#    total_loss: 损失函数 loss().
#    global_step: 全局训练次数（每批次训练一次step加1）
#  Returns:
#    train_op: 返回训练函数
def train(total_loss, global_step):
    # 训练次数=总样本数/每批次样本数
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # 学习率更新的步长，当global_step达到decay_steps的倍数时，学习率开始更新
    # 批量训练大小batch_size必须大于衰减步长，最好是NUM_EPOCHS_PER_DECAY的倍数，否则样本全部训练完了也不会更新
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 构建学习率阶梯下降训练模型，随着训练次数不断加大，学习率下降，不像mnist，学习率是固定的
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # 生成loss值的移动平均数统计量
    loss_averages_op = _add_loss_summaries(total_loss)

    # 梯度下降优化器，来优化和计算损失函数
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op

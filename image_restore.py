# coding: utf-8
import tensorflow as tf
import os
import scipy.misc
from tensorflow.python.platform import gfile

def distored_images(reshaped_image):
    height = 24
    width = 24
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)
    return float_image

def cifar102image(lines):
    # 创建文件名 list
    for i in range(1, 6):
        filenames = [os.path.join('cifar10_data/cifar-10-batches-bin', 'data_batch_%d.bin' % i)]

    # 使用 tensorflow 将文件名 list 转成队列（queue）
    filename_queue = tf.train.string_input_producer(filenames)

    # 标签占一个字节
    label_bytes = 1
    # 图片尺寸 32x32x3
    height = 32
    width = 32
    depth = 3

    # 一张图片字节数
    image_bytes = height * width * depth

    # 一帧数据包含一字节标签和 image_bytes 图片
    record_bytes = label_bytes + image_bytes

    # 创建固定长度的数据 reader
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)

    # 读出的 value 是 string，现在转换为 uint8 型的向量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 第一字节表示 标签值，我们把它从 uint8 型转成 int32 型
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的就是图片的数据了，我们把它的形状由 [深度*高(长)*宽] 转换成 [深度，高(长)，宽]
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [depth, height, width])

    # 将图片从 [深度，高(长)，宽] 转成 [高(长)，宽, 深度] 形状
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    reshaped_image = tf.cast(uint8image, tf.float32)
    distored_image = distored_images(reshaped_image)
    with tf.Session() as sess:
        #需要调用tf.train.start_queue_runners函数，否则数据流图将一直挂起
        # 在我们使用tf.train.string_input_producer创建文件名队列后，
        # “停滞状态”的，也就是说，我们文件名并没有真正被加入到队列中，此时如果我们开始计算，因为内存队列中什么也没有，计算单元就会一直等待，
        # 导致整个系统被阻塞。使用tf.train.start_queue_runners之后，才会启动填充队列的线程，这时系统就不再“停滞”。
        # 此后计算单元就可以拿到数据并进行计算，整个程序也就跑起来了。
        threads = tf.train.start_queue_runners(sess=sess)
        #sess.run(tf.initialize_all_variables())
        for i in range(10):
            _key, _label, image_array, distored_image_array = sess.run([key, label, reshaped_image, distored_image])
            scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.%s.jpg' % (i, lines[_label[0]]))
            scipy.misc.toimage(distored_image_array).save('cifar10_data/raw/%d.%s.distored.jpg' % (i, lines[_label[0]]))

def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x.replace("\n", "") if x != "\n" else x for x in f.readlines()]
        print(lines)
        return lines

if __name__ == "__main__":
    lines = load_CIFAR_Labels("./cifar10_data/cifar-10-batches-bin/batches.meta.txt")
    row_dir = 'cifar10_data/raw'
    if gfile.Exists(row_dir):
        gfile.DeleteRecursively(row_dir)
    gfile.MakeDirs(row_dir)
    cifar102image(lines)
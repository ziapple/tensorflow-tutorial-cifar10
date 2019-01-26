# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import img_input

class CIFAR10InputTest(tf.test.TestCase):

  def testSimple(self):
    filename = os.path.join('img_bin', "img_eval.bin")

    with self.test_session() as sess:
      q = tf.FIFOQueue(99, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      result = img_input.read_cifar10(q)

      for i in range(3):
        key, label, uint8image = sess.run([result.key, result.label, result.uint8image])
        self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))

      #with self.assertRaises(tf.errors.OutOfRangeError):
      #  sess.run([result.key, result.uint8image])


if __name__ == "__main__":
  tf.test.main()
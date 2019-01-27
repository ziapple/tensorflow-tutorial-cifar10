import tensorflow as tf
import numpy as np

def top_k_test():
    pred = np.array([[0.26271889,0.03923527,0.39857501,0.01027663,0.03053856,0.39669219,0.0447074,0.35643636,0.98240537,0.10904678]])
    print(pred)
    with tf.Session() as sess:
        topk = sess.run(tf.nn.top_k(pred, 1))
        print(topk)
        intopk = sess.run(tf.nn.in_top_k(pred, [8], 1))
        print(intopk)


if __name__ == '__main__':
    top_k_test()
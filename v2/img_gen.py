# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os

imgdir = './img_input'
tmpdir = './img_bin/tmp'

# 读取标签文件
def load_labels(filename):
    with open(filename, 'rb') as f:
        lines = [x.replace("\n", "") if x != "\n" else x for x in f.readlines()]
        return lines

def make():
    lines = load_labels("../cifar10_data/cifar-10-batches-bin/batches.meta.txt")
    i = 0
    for file in os.listdir(imgdir):
        label = int(file.split('-')[1].split('.')[0])
        img = Image.open(os.path.join(imgdir, file))
        # 压缩成32*32
        newimg = img.resize((32,32))
        newfilename = '%d.%s-%s.jpg' % (i, lines[label], label)
        newimg.save(os.path.join(tmpdir, newfilename))
        i += 1
    gen()

# 生成tmp文件夹下的cifar10格式文件
def gen():
    records = []
    for file in os.listdir(tmpdir):
        label = int(file.split('-')[1].split('.')[0])
        img = Image.open(os.path.join(tmpdir, file))
        img_arr = np.array(img)
        record = _gen_record(label, img_arr)
        records.append(record)

    contents = b"".join([_record for _record in records])
    filename = os.path.join('../cifar10_data', "img_eval.bin")
    open(filename, "wb").write(contents)


# 根据图片生成cifar10一条记录
def _gen_record(label, img_arr):
    b = bytearray()
    b.append(label)
    for k in range(3):
        for i in range(32):
            for j in range(32):
                b.append(img_arr[i, j, k])
    return bytes(b)

if __name__ == '__main__':
    make()
#! /usr/bin/python3

import os
import cv2
import tensorflow as tf
from random import shuffle


root = "./data"



'''
cut and paste 10% to sum/test, 2 doc-real&fake
'''

def convert_to_tfrecord(path_label, tfrecords_name):
    count = len(path_label)

    TFwriter = tf.python_io.TFRecordWriter(tfrecords_name)

    for path, label in path_label:
        label = int(label)

        print (path, label)

        img = cv2.imread(path)
        height, width, channels = img.shape
        imgRaw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                    "height": tf.train.Feature(int64_list = tf.train.Int64List(value=[height])),
                    "width": tf.train.Feature(int64_list = tf.train.Int64List(value=[width])),
                    "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
                }) )
        TFwriter.write(example.SerializeToString())

    TFwriter.close()


def main():
    fin = open('data/path.txt', 'r')
    path_label = [line.strip().split(' ') for line in fin.readlines()]
    shuffle(path_label)
    total = len(path_label)
    convert_to_tfrecord(path_label[:int(total * 0.9)], "train.tfrecords")
    convert_to_tfrecord(path_label[int(total * 0.9):], "test.tfrecords")    

if __name__ == '__main__':
    main()

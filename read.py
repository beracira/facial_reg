#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import cv2
from skimage.viewer import ImageViewer

tfrecords_filename = 'train.tfrecords'

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['img'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    
    image = tf.reshape(image, image_shape)
    
    return image

def main():
  filename_queue = tf.train.string_input_producer(
      [tfrecords_filename])

  image = read_and_decode(filename_queue)

  init_op = tf.initialize_all_variables()

  with tf.Session()  as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(3):
      img = sess.run([image])
      img = img[0]
      print (img[0])
      viewer = ImageViewer(img)
      viewer.show()

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  main()

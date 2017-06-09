#! /usr/bin/python3

import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
BATCH_SIZE = 10
TOTAL = 118
TRAIN = 100
TEST = TOTAL - TRAIN

fin = open('data/path.txt', 'r')
lines = [line.strip() for line in fin]

filename_queue = tf.train.string_input_producer(lines)

filename, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")
image_file = tf.read_file(filename)

image = tf.image.decode_jpeg(image_file, channels=3)
image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
image = tf.reshape(image, [IMAGE_SIZE * IMAGE_SIZE * 3])
label = tf.reshape(label, [1])
# image = tf.reshape(label, tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))

# image_batch, label_batch = tf.train.batch(
#       [image, label], batch_size=BATCH_SIZE, capacity=200)#, min_after_dequeue=100)


x = tf.placeholder(tf.float32, [IMAGE_SIZE * IMAGE_SIZE * 3])
W = tf.Variable(tf.zeros([IMAGE_SIZE * IMAGE_SIZE * 3]))
b = tf.Variable(tf.zeros([1]))
y = tf.reduce_sum(tf.multiply(x, W)) + b
# y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [1])

cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)



  # image_batch = tf.train.batch(image_batch, BATCH_SIZE)
  # label_batch = tf.train.batch(labels, BATCH_SIZE)

  # Train
  for _ in range(TRAIN):
    image_, label_ = sess.run([image, label])
    label_ = [int(label_[0].decode())]
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: image_, y_: label_})


   # Test trained model
  correct = 0
  for _ in range(TEST):
    image_, label_ = sess.run([image, label])
    label_ = [int(label_[0].decode())]
    # correct_prediction = tf.equal(tf.argmax(y, 0), tf.argmax(y_, 0))
    # print (label_, sess.run([y, y_]))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predict, ans = sess.run([y, y_], feed_dict={x: image_,
                                        y_: label_})
    if (predict >= 0): predict = 1
    else: predict = 0
    if (predict == ans): correct += 1
    print (predict, ans)
  print ("Acc: ", correct / TEST)

  coord.request_stop()
  coord.join(threads)

  # Finish off the filename queue coordinator.



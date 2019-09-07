import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

print(22)

IMG_H = 100
IMG_W = 100
IMG_C = 3

NUM_CLASS = 3

APPLE = 0
BANANA = 1
ORANGE = 3

trainList, testList = list(), list()
folderPath = "./폴수학학교/2019 가을학기/동아리/Brand_Logo_Classification_Data/"

with open(folderPath + "train.txt") as f:
    for line in f:
        tmp = line.strip().split()
        trainList.append(tmp[0], tmp[1])

with open(folderPath +  "test.txt") as f:
    for line in f:
        tmp = line.strip().split()
        testList.append(tmp[0], tmp[1])

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_W * IMG_H * IMG_C])
    Y = tf.placeholder(tf.int32, [None])
    dim = 2048

with tf.variable_scope('MLP'):
    net = tf.layers.dense(X, dim)

    for i in range(4):
        net = tf.layers.dense(net, dim / 2 ** i, activation = tf.nn.relu)

    out = tf.layers.dense(net, NUM_CLASS)

with tf.variable_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y, logits = out))
    train = tf.train.AdamOptimizer(0.001).minimize(loss)
    saver = tf.train.Saver()

def ReadImage(path):
    img = plt.imread(path)
    img = np.reshape(img, (IMG_H * IMG_W * IMG_C))

    return img

def Batch(path, batchSize):
    img, label, paths = list(), list(), list()

    for _ in range(batchSize):
        img.append(ReadImage(path[0][0]))
        label.append(int(path[0][1]))
        path.append(paths.pop(0))

        return img, label

batchSize = 50

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5000):
        batchData, batchLabel = Batch(trainList, batchSize)
        l = sess.run([train, loss], feed_dict = {X: batchData, Y: batchLabel})
        print(i, l)
        saver.save(sess, 'logs/model.ckpt', global_step = i + 1)
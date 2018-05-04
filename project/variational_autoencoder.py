""" Variational Auto-Encoder Example.

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.

References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
Update: Sapumal Ahangama on VAE for 1st and 2nd order proximity of network and shared data 
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import norm
import tensorflow as tf
from gensim.models.doc2vec import Doc2Vec
import gensim
import gensim.utils as ut
from collections import namedtuple
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.0001
#num_steps = 30000
num_epochs = 1
batch_size = 1000

#input params
n = 30422 #nodes in the network
m = 300 #word2vec embedding

alpha = 0.2

# Network Parameters
input_dim = (n + m) 
hidden_dim1 = 5000
hidden_dim2 = 1000
latent_dim = 300

#Union module network + text data
def trainDoc2Vec(doc_list=None, buildvoc=1, passes=20, dm=0, size=100, dm_mean=0, window=8, hs=1, negative=5, min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window, hs=hs, negative=negative, min_count=min_count, workers=workers) #PV-DBOW

    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):

        shuffle(doc_list)  # shuffling gets best results
        model.train(doc_list)

    return model

NetworkSentence=namedtuple('NetworkSentence', 'words tags index')
def readNetworkData(dir, stemmer=0): #dir, directory of network dataset
    allindex={}
    alldocs = []
    labelset = set()
    with open(dir + '/new_ab.txt') as f1:
        for l1 in f1:
            #tokens = ut.to_unicode(l1.lower()).split()
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()

            words = tokens[1:]
            tags = [tokens[0]] # ID of each document, for doc2vec model
            index = len(alldocs)
            allindex[tokens[0]] = index # A mapping from documentID to index, start from 0
            alldocs.append(NetworkSentence(words, tags, index))

    return alldocs, allindex,

#Load the dataset
directory = 'dataset/dblp_new'
network = np.load("dataset/dblp_new/dblp.npy")
group = np.load("dataset/dblp_new/label.npy")

alldocs, allsentence = readNetworkData(directory)
doc_list = alldocs[:]  # for reshuffling per pass
tridnr_model = trainDoc2Vec(doc_list, workers=0, size=m, dm=0, passes=10, min_count=3)
vecs = [tridnr_model.docvecs[ut.to_unicode(str(j))] for j in range(n)]

union = np.concatenate([network,vecs],axis=1)

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([input_dim, hidden_dim1])),
    'encoder_h2': tf.Variable(glorot_init([hidden_dim1, hidden_dim2])),
    'z_mean1': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'z_std1': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'z_mean2': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'z_std2': tf.Variable(glorot_init([hidden_dim2, latent_dim])),
    'decoder_h2': tf.Variable(glorot_init([(latent_dim*2), hidden_dim2])),
    'decoder_h1': tf.Variable(glorot_init([hidden_dim2, hidden_dim1])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim1, input_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim1])),
    'encoder_b2': tf.Variable(glorot_init([hidden_dim2])),
    'z_mean1': tf.Variable(glorot_init([latent_dim])),
    'z_std1': tf.Variable(glorot_init([latent_dim])),
    'z_mean2': tf.Variable(glorot_init([latent_dim])),
    'z_std2': tf.Variable(glorot_init([latent_dim])),
    'decoder_b2': tf.Variable(glorot_init([hidden_dim2])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim1])),
    'decoder_out': tf.Variable(glorot_init([input_dim]))
}

# Building the encoder
adj_matrix = tf.placeholder(tf.float32, shape=[None, None])
input_data = tf.placeholder(tf.float32, shape=[None, input_dim])
encoder = tf.matmul(input_data, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
encoder = tf.matmul(encoder, weights['encoder_h2']) + biases['encoder_b2']
encoder = tf.nn.tanh(encoder)
z_mean1 = tf.matmul(encoder, weights['z_mean1']) + biases['z_mean1']
z_std1 = tf.matmul(encoder, weights['z_std1']) + biases['z_std1']
z_mean2 = tf.matmul(encoder, weights['z_mean2']) + biases['z_mean2']
z_std2 = tf.matmul(encoder, weights['z_std2']) + biases['z_std2']

# Sampler: Normal (gaussian) random distribution
eps1 = tf.random_normal(tf.shape(z_std1), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z1 = z_mean1 + tf.exp(z_std1 / 2) * eps1

eps2 = tf.random_normal(tf.shape(z_std2), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z2 = z_mean2 + tf.exp(z_std2 / 2) * eps2

con = tf.concat([z1, z2], 1) 

# Building the decoder (with scope to re-use these layers later)
decoder = tf.matmul(con, weights['decoder_h2']) + biases['decoder_b2']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)

    # KL Divergence loss
    kl_div_loss1 = 1 + z_std1 - tf.square(z_mean1) - tf.exp(z_std1)
    kl_div_loss1 = -0.5 * tf.reduce_sum(kl_div_loss1, 1)

    kl_div_loss2 = 1 + z_std2 - tf.square(z_mean2) - tf.exp(z_std2)
    kl_div_loss2 = -0.5 * tf.reduce_sum(kl_div_loss2, 1)

    #1st order loss
    Diag = tf.diag(tf.reduce_sum(adj_matrix, 1))
    Lap = Diag - adj_matrix ##L is laplacion-matriX
    fst_loss = 0 #2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(con),Lap),con)) ##ISSUE WITH THIS LINE

    return tf.reduce_mean(encode_decode_loss + kl_div_loss1 + kl_div_loss2 + (alpha * fst_loss))

loss_op = vae_loss(decoder, input_data)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_epochs+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, _ = mnist.train.next_batch(batch_size)
        for start, end in zip(range(0, len(network), batch_size), range(batch_size, len(network) + 1, batch_size)):
            batch_x = union[start:end] 

            # Train
            feed_dict = {input_data: batch_x, adj_matrix: network}
            _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)

            print('Epoch %i, Loss: %f' % (i, l))

    # Testing
    print("Testing")
    embed = sess.run(con, feed_dict={input_data: union})
    
    for t in range(4):
        train_size = (t * 0.2) + 0.1
        for l in range(10):
            random_state = l
            train, test, y_train, y_test = train_test_split(embed, group, train_size=train_size, random_state=random_state)
            classifier = LinearSVC()
            classifier.fit(train, y_train)
            y_pred = classifier.predict(test)
            macro_f1 = f1_score(y_test, y_pred, pos_label=None, average='macro')
            micro_f1 = f1_score(y_test, y_pred, pos_label=None, average='micro')
            print('Classification macro_f1=%f, micro_f1=%f' % (macro_f1, micro_f1)) 

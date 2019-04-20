import tensorflow as tf
from datetime import datetime
import numpy as np
import pickle
import os
import pdb
from time import time

# Ignore TF warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maxlength = 20
batch_size = 1
nhidden = 1024
down_proj_dim = 512
embedding_size = 100
vocabsize = 20004
SUBSET = 0
NEPOCHS = 1
BUFFER_SIZE = 10000
LOG_EVERY = 200
CHECKPT_PATH = '/home/songbird/Desktop/'
DATA_PATH = './data/'


def load_data(subset = SUBSET):
    with open(DATA_PATH+'training_data.pickle','rb') as f:
        d = pickle.load(f)
    data = np.array(d['data'])
    V = d['vocabulary']
    del d
    # random subset 
    if subset > 0:
        idx = np.random.choice(len(data), size = subset)
        data = data[idx]
    return data, V

## MAIN BODY ##
start = time()
# load data
# setup the dataset
_, V = load_data()
vocabsize = len(V.keys())
continuation_data = np.load(DATA_PATH + 'continuation_data.npy')
continuation_data = continuation_data[:1]

def to_batches(data, shuffle = True, pad = False):
    nsamps = len(data)
    nbatches = nsamps // batch_size
    rem = batch_size - (nsamps - nbatches*batch_size)
    if rem > 0 and pad:
        # concatenate zero vectors to data
        padd = np.zeros((rem, maxlength), dtype='int32')
        data = np.concatenate([data, padd],axis=0)
        nsamps = len(data)
        nbatches = nsamps // batch_size
        assert nbatches*batch_size==nsamps, 'Still not equal!'
        
    if shuffle:
        np.random.shuffle(data)
    data_batches = [None for _ in range(nbatches)]
    k = 0
    for i in range(nbatches):
        data_batches[i] = data[k : k + batch_size]
        k += batch_size
        
    return data_batches, rem

data_batches,_ = to_batches(continuation_data, shuffle = False, pad = False)

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("inputs"):
        x = tf.placeholder(shape = (1), dtype=tf.int32, name='x')
        #y = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='y')
        
    with tf.name_scope("params"):
        word_embeddings = tf.get_variable(name='word_embeddings', shape=[vocabsize, embedding_size], 
                                         trainable=True)
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units = nhidden, use_peepholes = False, 
                                  initializer = tf.contrib.layers.xavier_initializer(),
                                         name = 'rnn')
        W1 = tf.get_variable(name='down_proj', shape=[nhidden, down_proj_dim],
                                    initializer = tf.contrib.layers.xavier_initializer(),
                           trainable = True)
        b1 = tf.get_variable(name = 'down_bias', initializer = tf.zeros(down_proj_dim), trainable=True)
        W2 = tf.get_variable(name='linear_out', shape=[down_proj_dim, vocabsize],
                                    initializer = tf.contrib.layers.xavier_initializer(),
                           trainable = True)
        b2 = tf.get_variable(name = 'bias', initializer = tf.zeros(vocabsize), trainable=True)
        
    with tf.name_scope("state"):
        # initialize the rnncell state 
        rnn_state = tf.placeholder(shape=(None, nhidden), dtype = tf.float32, name = 'rnn_state')
        rnn_cell_state = tf.placeholder(shape=(None, nhidden), dtype = tf.float32, name = 'rnn_cell_state')
        h = (rnn_state, rnn_cell_state)
        
    with tf.name_scope("prediction"):
        E = tf.nn.embedding_lookup(word_embeddings, x)
        
        # map through lstm
        o, h = rnncell(E, h)
        # map output to score over vocabulary
        z1 = tf.matmul(o, W1) + b1
        z2 = tf.matmul(z1, W2) + b2
        #probability = tf.nn.softmax(z2)
        # get index of max
        output = tf.argmax(z2, axis = -1)
        output = tf.squeeze(output)
    #with tf.name_scope("loss"):
    #    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
    #                                                                         logits = output),
    #                          name='loss')
        
    #with tf.name_scope("optimize"):
    #    optimizer = tf.train.AdamOptimizer()
    #    grads_and_variables = optimizer.compute_gradients(loss)
        #grads_and_variables = list(zip(*grads_and_variables))
        #grads = grads_and_variables[0]
    #    grads = [g for g,v in grads_and_variables
    #    variables = [v for g,v in grads_and_variables]
    #    clipped_grads = tf.clip_by_global_norm(grads, clip_norm = 5)
        #print([tf.shape(c) for c in clipped_grads])t
    #    clipped_grads = clipped_grads[0]
    #    train_step = optimizer.apply_gradients(zip(clipped_grads, variables))

        
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    saver.restore(session, save_path = CHECKPT_PATH + '/model_checkpt_exC/'+'final_model.ckpt')
    # Training loop
    all_generated_sentences = []
    for i, batch_sentence in enumerate(data_batches):
        not_done = True
        t = 0
        output_sequence = []
        while not_done:
            if t==0:
                h, c = session.run([rnn_state, rnn_cell_state], 
                                   feed_dict={rnn_state: np.zeros((batch_size, nhidden), dtype='float32'),
                                             rnn_cell_state: np.zeros((batch_size, nhidden), dtype='float32') })
                
            if batch_sentence[:,t] == V['<pad>'][0]:
                # use last generated word
                output_word = session.run([output],
                                          feed_dict = {x: output_sequence[t-1],
                                                      rnn_state: h,
                                                      rnn_cell_state: c})
                output_sequence.append(output_word)
            else:
                output_word = session.run([output], feed_dict = {x: batch_sentence[:,t],
                                                                rnn_state: h,
                                                                rnn_cell_state: c})
                output_sequence.append(output_word)
                
            if output_word == V['<eos>'][0]:
                not_done = False
            t += 1
            if t == maxlength:
                not_done = False
        
        # convert output_sequences to strings
        out = []
        for w in output_sequence:
            for word, value in V.items():
                if w[0] == value[0]:
                    out.append(word)
                    break
        all_generated_sentences.append(out)
        
        if i % LOG_EVERY == LOG_EVERY-1:
            map(print, out)
        
    with open(CHECKPT_PATH+'ex2_generated.txt', 'w') as f:
        for i in range(len(all_generated_sentences)):
            s = all_generated_sentences[i]
            for t in range(len(s)):
                f.write(str(s[t])+' ')
            f.write('\n')
            
end = time()
print('\n ... DONE! time taken = %5d secs ...' % (end - start))
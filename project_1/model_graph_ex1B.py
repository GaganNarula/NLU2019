from load_embedding import *
from datetime import datetime
import numpy as np
import pickle
import os
import pdb
from time import time

# Ignore TF warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maxlength = 30
batch_size = 64
nhidden = 512
embedding_size = 100
vocabsize = 20004
SUBSET = 0
NEPOCHS = 1
BUFFER_SIZE = 10000
LOG_EVERY = 200
path2emb = './data/wordembeddings-dim100.word2vec'
CHECKPT_PATH = '/home/songbird/Desktop/'
DATA_PATH = './data/'
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = CHECKPT_PATH + "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)


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

def model_step(sentence_batch, maxlen = 30, bs = 64):
    # initialize the rnncell state 
    rnn_state = tf.zeros((batch_size, nhidden))
    rnn_cell_state = tf.zeros((batch_size, nhidden))
    h = (rnn_state, rnn_cell_state)
    output = []
    # go through the sentence word by word until the second last word (last is <eos>)
    for i in range(maxlen-1):
        # get the embedding vector for each sentence in batch
        E = tf.nn.embedding_lookup(word_embeddings, sentence_batch[:, i])
        # map through lstm
        o, h = rnncell(E, h)
        # map output to score over vocabulary
        z = tf.matmul(o, W) + b
        output.append(z)
    return tf.stack(output, axis=1)

def perplexity(label, yhat, V, bs = 64):
    # go through sequence sum up
    batch_perplexity = np.zeros(bs)
    for j in range(bs):
        perp = 0.
        cnt = 0.
        for t in range(maxlength-1):
            if label[j,t] == V['<pad>'][0]:
                continue
            else:
                tmp = yhat[j,t,label[j,t]]
                perp += np.log2(tmp)
                cnt += 1.
        perp = perp / cnt
        batch_perplexity[j] = (2)**(-perp)
    return batch_perplexity

## MAIN BODY ##
start = time()
# load data
# setup the dataset
data, V = load_data()
vocabsize = len(V.keys())
test_data = np.load(DATA_PATH + 'test_data.npy')

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

data_batches,_ = to_batches(data)
nsampstest = len(test_data)
test_data_batches, rem = to_batches(test_data, shuffle = False, pad = True)

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("inputs"):
        x = tf.placeholder(shape=(None, maxlength), dtype=tf.int32, name='x')
        y = tf.placeholder(shape=(None, maxlength-1), dtype=tf.int32, name='y')
        
    with tf.name_scope("params"):
        word_embeddings = tf.get_variable(name='word_embeddings', shape=[vocabsize, embedding_size], 
                                         trainable=True)
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units = nhidden, use_peepholes = False, 
                                  initializer = tf.contrib.layers.xavier_initializer(),
                                         name = 'rnn')
        W = tf.get_variable(name='linear_out', shape=[nhidden, vocabsize],
                                    initializer = tf.contrib.layers.xavier_initializer(),
                           trainable = True)
        b = tf.get_variable(name = 'bias', initializer = tf.zeros(vocabsize), trainable=True)
        
    with tf.name_scope("prediction"):
        output = model_step(x)
        
    
    with tf.name_scope("get_softmax"):
        probability = tf.nn.softmax(output)
        
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                             logits = output),
                              name='loss')
        
    with tf.name_scope("optimize"):
        optimizer = tf.train.AdamOptimizer()
        grads_and_variables = optimizer.compute_gradients(loss)
        grads = [g for g,v in grads_and_variables]
        variables = [v for g,v in grads_and_variables]
        clipped_grads = tf.clip_by_global_norm(grads, clip_norm = 5)
        clipped_grads = clipped_grads[0]
        train_step = optimizer.apply_gradients(zip(clipped_grads, variables))

        
    saver = tf.train.Saver()

loss_summary = tf.summary.scalar("LOSS", loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    load_embedding(session, V, word_embeddings, path2emb, embedding_size, vocabsize)
    # Training loop
    for epoch in range(NEPOCHS):
        train_loss = []
        for i, sentence in enumerate(data_batches):
            labels = sentence[:,1:]
            if i % LOG_EVERY == LOG_EVERY-1:
                print('... loss at batch %d is = %f ... '%(i, train_loss[i-1]))
                summary_str = loss_summary.eval(feed_dict = {x: sentence, y: labels})
                step = i
                file_writer.add_summary(summary_str, step)
                
            _, loss_, _ = session.run([train_step, loss, output], feed_dict = {x: sentence, y: labels})
            train_loss.append(loss_)
            
            
        # checkpoint model
        save_path = saver.save(session, CHECKPT_PATH + '/model_checkpt/'+'final_model.ckpt')


        # evaluate learned model
    sentence_perplexity = []
    for i, sentence in enumerate(test_data_batches):
        labels = sentence[:, 1:]
        p = session.run(probability, feed_dict = {x: sentence, y: labels})
        sp = perplexity(labels, p, V, labels.shape[0])
        sentence_perplexity.append(sp)
    # remove "rem" vectors
    sentence_perplexity = np.concatenate(sentence_perplexity, axis=0)
    sentence_perplexity = sentence_perplexity[:nsampstest]
    assert len(sentence_perplexity)==nsampstest, 'unequal lengths!!'
    with open(CHECKPT_PATH+'test_perpB.txt', 'w') as f:
        for i in range(len(sentence_perplexity)):
            f.write(str(sentence_perplexity[i])+'\n')
end = time()
print('\n ... DONE! time taken = %5d secs ...' % (end - start))
file_writer.close()
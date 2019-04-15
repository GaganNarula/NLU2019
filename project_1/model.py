import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import pickle
import pdb

max_length = 30
batch_size = 64
BUFFER_SIZE = 10000
# make this 0 if you want to process whole dataset, else this is the number of sentences
SUBSET = 1000
NEPOCHS = 1
# afte how many batches do you want to print the loss?
LOG_EVERY = 2
CHECKPT_PATH = ''

def load_data(subset = SUBSET):
    with open(CHECKPT_PATH+'training_data.pickle','rb') as f:
        d = pickle.load(f)
    data = np.array(d['data'])
    V = d['vocabulary']
    del d
    # random subset 
    if subset > 0:
        idx = np.random.choice(len(data), size = subset)
        data = data[idx]
    return data, V

def define_model(embedding_size = 512, vocabsize = 20004):
    word_embeddings = tf.get_variable('word_embeddings', [vocabsize, embedding_size], trainable=True)
    rnncell = tf.nn.rnn_cell.LSTMCell(num_units = 512, use_peepholes = False, 
                                  initializer = tf.contrib.layers.xavier_initializer())
    linear_out = tf.layers.Dense(units=vocabsize)
    return rnncell, linear_out, word_embeddings


class LM(object):
    def __init__(self, maxlen = 30, embedding_size = 512, vocabsize = 20004):
        self.word_embeddings = tf.get_variable('word_embeddings', [vocabsize, embedding_size], trainable=True)
        self.rnncell = tf.nn.rnn_cell.LSTMCell(num_units = 512, use_peepholes = False, 
                                  initializer = tf.contrib.layers.xavier_initializer())
        self.linear_out = tf.layers.Dense(units=vocabsize)
        self.maxlen = maxlen
        
    def __call__(self, sentence_batch, maxlen = 30):
        # initialize the rnncell state 
        rnn_state = tf.zeros([sentence_batch.shape[0], self.rnncell.state_size[0]])
        rnn_cell_state = tf.zeros([sentence_batch.shape[0], self.rnncell.state_size[0]])
        h = (rnn_state, rnn_cell_state)
        output = []
        # go through the sentence word by word until the second last word (last is <eos>)
        for i in range(self.maxlen-1):
            # get the embedding vector for each sentence in batch
            E = [self.word_embeddings[w, :] for w in sentence_batch[:,i]]
            E = tf.stack(E, axis = 0)
            # map through lstm
            o, h = self.rnncell(E, h)
            # map output to score over vocabulary
            output.append(self.linear_out(o))
        return tf.stack(output, axis=1)
    
    
def model_step(maxlen = 30):
    # initialize the rnncell state 
    rnn_state = tf.zeros([sentence_batch.shape[0], self.rnncell.state_size[0]])
    rnn_cell_state = tf.zeros([sentence_batch.shape[0], self.rnncell.state_size[0]])
    h = (rnn_state, rnn_cell_state)
    output = []
    # go through the sentence word by word until the second last word (last is <eos>)
    for i in range(maxlen-1):
        # get the embedding vector for each sentence in batch
        E = [self.word_embeddings[w, :] for w in sentence_batch[:,i]]
        E = tf.stack(E, axis = 0)
        
        # map through lstm
        o, h = self.rnncell(E, h)
        
        # map output to score over vocabulary
        output.append(self.linear_out(o))
    return tf.stack(output, axis=1)


def loss(labels, prediction):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=prediction)
    s = tf.reduce_mean(l)
    return s


def train(dataset, model, optimizer, vocab = [], nepochs = 1, log_every = 100):
    per_epoch_loss = []
    for n in range(nepochs):
        per_batch_loss = []
        for (i, sentence_batch) in enumerate(dataset):
            with tf.GradientTape() as t:
                t.watch(sentence_batch)
                output = model(sentence_batch)
                # the label is simply the shifted input
                label = sentence_batch[:,1:]
                # compute loss
                L = loss(label, output)

                # create variable list
                rw = model.rnncell.weights
                lo = model.linear_out.weights
                we = model.word_embeddings
                variables = []
                for k in range(len(rw)):
                    variables.append(rw[k])
                for l in range(len(lo)):
                    variables.append(lo[l])
                variables.append(we)
                #variables = [rnncell.weights, linear_out.weights, word_embeddings]

                grads = t.gradient(L, variables)
                # clip gradients
                grads = tf.clip_by_global_norm(grads, clip_norm = 5) 
                # remove last element
                grads = grads[0]

                optimizer.apply_gradients(zip(grads, variables))
                
                per_batch_loss.append(L)
                
                if i%log_every == log_every-1:
                    print('\n ... loss after %d batches = %f ... '%(i+1, per_batch_loss[i]))
                    #if i==log_every-1:
                        #P = eval(model, vocab)
                        #print('evaluation perplexity = %5d'%(P.mean()))
        per_epoch_loss.append(np.array(per_batch_loss))
    return model, per_epoch_loss


def eval(model, V):
    # load test data
    test_data = np.load('./data/test_data.npy')
    slices = tf.data.Dataset.from_tensor_slices(test_data)
    dataset = slices.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    sentence_perplexity = []
    for (i, sentence_batch) in enumerate(dataset):
        output = model(sentence_batch)
        # the label is simply the shifted input
        label = sentence_batch[:,1:]
        pred = tf.nn.softmax(output)
        sp = perplexity(label, pred, V)
        sentence_perplexity.append(sp)
    return np.concatenate(sentence_perplexity, axis=0)


def perplexity(label, pred, V):
    # go through sequence sum up
    batch_perplexity = np.zeros(batch_size)
    for j in range(batch_size):
        perp = 0.
        cnt = 0.
        for t in range(max_length-1):
            if label[j,t] == V['<pad>'][0]:
                continue
            else:
                perp += np.log2(pred[j,t,label[j,t]].numpy())
                cnt += 1.
        perp = perp / cnt
        batch_perplexity[j] = tf.pow(2, -perp)
    return batch_perplexity
        
    
def main():
    # setup the dataset
    data, V = load_data()
    print('\n ... data loaded, number of sentences = %d ... '%(len(data)))
    slices = tf.data.Dataset.from_tensor_slices(data)
    dataset = slices.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    # setup model
    #rnncell, linear_out, word_embeddings = define_model()
    model = LM()
    print('\n ... model constructed ... ')
    # setup optimizer
    optimizer = tf.train.AdamOptimizer()
    # train
    model, per_epoch_loss = train(dataset, model, optimizer, V, NEPOCHS, LOG_EVERY)
    # save model
    print('\n ... saving model ... ')
    saver = tfe.Saver([model.rnncell.weights, model.linear_out.weights,
                       model.word_embeddings])
    saver.save(CHECKPT_PATH+'model.ckpt')
    # evaluate model
    sentence_perplexity = eval(model, V)
    # save perplexity
    with open(CHECKPT_PATH+'test_perp.txt', 'w') as f:
        for i in range(len(sentence_perplexity)):
            f.write(str(s[i])+'\n')
            
if __name__=='__main__':
    main()
    
    
    
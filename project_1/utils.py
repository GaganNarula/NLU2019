from itertools import chain
from collections import Counter
import numpy as np
import pickle

MAXLEN_WITHOUT_BOS_EOS = 28

def readLines(path_to_file):
    with open(path_to_file,'r') as f:
        data = []
        for lines in f:
            lines = lines.split(' ')
            if len(lines) <= MAXLEN_WITHOUT_BOS_EOS:
                data.append(lines)
    return data


def makeVocabulary(data, freq_limit = 20000):
    nsents = len(data)
    data_long = list(chain.from_iterable(data))
    # find all unique strings 
    # and iterate over freq_limit most common one's
    unique_n_count = Counter(data_long)
    vocabulary = {}
    for i,t in enumerate(unique_n_count.most_common(freq_limit)):
        vocabulary[t[0]] = (i,t[1]) # i is the index of this word
    # now add the special symbols 
    vocabulary['<bos>'] = (i+1, nsents)
    vocabulary['<pad>'] = (i+2, 1)
    vocabulary['<eos>'] = (i+3, nsents)
    vocabulary['<unk>'] = (i+4, 1)
    return vocabulary


def map2Index(sentence, vocabulary):
    indx_out = [None for _ in range(len(sentence))]
    for (i,w) in enumerate(sentence):
        if w in vocabulary:
            indx_out[i] = vocabulary[w][0]
        else:
            indx_out[i] = vocabulary['<unk>'][0]
    return np.array(indx_out)


def padSentence(sentence, maxlen = 30):
    L = len(sentence)
    if L < maxlen:
        for i in range(maxlen - L):
            sentence.append('<pad>')
    return sentence


def preprocessSentence(sentence, vocabulary, maxlen = 30):
    s = sentence.copy()
    # first add special <bos> symbol to beginning
    s.insert(0,'<bos>')
    # pad to maxlen-1, (-1 because of <eos> at the end)
    s = padSentence(s, maxlen-1)
    # add <eos> to the end
    s.append('<eos>')
    # get index and return sentence
    indexed = map2Index(s, vocabulary)
    return indexed, s


def createDataset(path_to_file, save_to):
    # load data
    data = readLines(path_to_file)
    maxlen = [len(s) for s in data]
    maxlen = max(maxlen)
    print('... loaded data ...')
    print('... total numb of valid sentences = %d ...' % (len(data)))
    print('... maximum sentence length = %d ...'%(maxlen))
    # make vocabulary
    V = makeVocabulary(data)
    print('... made vocabulary ...')
    # preprocess sentences
    data = [preprocessSentence(s, V) for s in data]
    data = np.array([d[0] for d in data])
    print('... processed sentences ... ')
    # data dict
    d = {'data': data, 'vocabulary': V}
    # pickle this  with protocol = 3
    with open(save_to, 'wb') as f:
        pickle.dump(d, f, protocol = 3)
        
        
def processTestDataset(path_to_test, save_to, V):
    # load data 
    data = readLines(path_to_test)
    # preprocess sentences
    data = [preprocessSentence(s, V) for s in data]
    data = np.array([d[0] for d in data])
    print('... processed sentences ... ')
    np.save(save_to, data)

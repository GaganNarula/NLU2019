from models import *
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type = str, help= 'path to data, required')
parser.add_argument('--path_to_modelfile', type = str, help = 'path to saved model')
parser.add_argument('--savepath', type = str, help = 'path to output')


def tokenize_sentence(tokenizer, sentence, cuda = False):
    tks = tokenizer.tokenize(s)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tks)
    tokens_tensor = torch.tensor([indexed_tokens])
    if cuda:
        tokens_tensor = tokens_tensor.to('cuda')
    return tokens_tensor

def embed_tokenized_sentence(model, tokens_tensor, to_numpy = False, to_cpu = False):
    with torch.no_grad():
        enc_layers,_ = model(tokens_tensor)
    out = enc_layers[-1].squeeze()
    if to_cpu:
        out = out.cpu()
    if to_numpy:
        out = out.numpy()
    return out

def load_bert(cuda = False):
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    if cuda:
        model.to('cuda')
    return model


def eval(sentence_tuple, rnn_model, embedding_model, tokenizer, cudaa = False):
    # first tokenize each input sentence in tuple
    x = []
    for i in range(4):
        tks = tokenize_sentence(tokenizer, sentence_tuple[i], cudaa)
        embedded_seq = embed_tokenized_sentence(embedding_model, 
                                                tokens_tensor, 
                                                to_numpy = False, to_cpu = False)
        x.append(embedded_seq)
    x = torch.cat(x, dim = 0)
    
    # concatenate with output and make two versions
    tks = tokenize_sentence(tokenizer, sentence_tuple[4], cudaa)
    embedded_seq = embed_tokenized_sentence(embedding_model, 
                                                tokens_tensor, 
                                                to_numpy = False, to_cpu = False)
    y1 = torch.cat([x, embedded_seq], dim = 0)
    
    tks = tokenize_sentence(tokenizer, sentence_tuple[5], cudaa)
    embedded_seq = embed_tokenized_sentence(embedding_model, 
                                                tokens_tensor, 
                                                to_numpy = False, to_cpu = False)
    y2 = torch.cat([x, embedded_seq], dim = 0)
    
    # get lengths
    lengths = [y1.size(0), y2.size(0)]
    maxlen = torch.max(lengths)
    order = torch.from_numpy(np.ascontiguousarray(np.flip(np.argsort(lengths.numpy()))))
    # pad both 
    y = torch.nn.utils.rnn.pad_sequence([y1 , y2], batch_first = True)
    # sort to decreasing order
    y = y[order]
    lengths = lengths[order]
    # model output
    with torch.no_grad():
        logp = rnn_model(y, lengths, maxlen)
    
    # reorder
    order = order.numpy().tolist()
    # make a decision
    lgp1 = -logp[order.index(0)].cpu().numpy()
    lgp2 = -logp[order.index(1)].cpu().numpy()
    idx = np.argmin(np.array([lgp1, lgp2]))
    predlabel = idx + 1 # the labels from StoryCloze are [1,2]
    return predlabel


def eval_sentences(sentence_tuples):
    N = len(sentence_tuples)
    Predlabels = np.zeros(N, dtype = int)
    for i in range(N):
        Predlabels[i] = eval(sentence_tuples.iloc[i,:])
    return Predlabels

def get_sentences(file):
    sents = []
    with open(file, 'r') as f:
        for line in f.readlines():
            sents.append(line)
    return sents

if __name__ == '__main__':
    args = parser.parse_args()
    
    data = pd.read_csv(args.datapath)
    print('\n ... loading model ...\n')
    rnn_model = rnn(nhidden=50, nlin1=600, num_layers = 2, bidirectional = True, dropout = 0.2)
    rnn_model.load_state_dict(torch.load(args.path_to_modelfile))
    rnn_model.eval()
    
    embedding_model = load_bert()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('\n ... evaluating model ....')
    predicted_labels = eval_sentences(data)
    print('\n ... writing output ...')
    df = pd.Dataframe(predicted_labels)
    df.to_csv(args.savepath, header = False)
    print('\n ... DONE ...')
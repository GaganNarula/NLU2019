import torch
import torch.nn as nn

class rnn(nn.Module):
    def __init__(self, ninput = 768, nhidden = 512, nlin1 = 768, num_layers = 1, bidirectional = False,
                dropout = 0):
        super(rnn, self).__init__()
        self.lin_in = nn.Linear(ninput, 400)
        self.gru = nn.GRU(input_size=400, hidden_size=nhidden, num_layers= num_layers,
                          batch_first = True, bidirectional=bidirectional, dropout = dropout)
        if bidirectional:
            self.lin1 = nn.Linear(1 * 2 * nhidden, nlin1)
            self.init_lin = nn.Linear(100, num_layers * 2 * nhidden) 
        else:
            self.lin1 = nn.Linear(1 * 1 * nhidden, nlin1)
            self.init_lin = nn.Linear(100, num_layers * 1 * nhidden) 
        self.lin2_mu = nn.Linear(nlin1, ninput)
        self.lin2_sig = nn.Linear(nlin1, ninput)
        self.splus = nn.Softplus()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.layernorm1= nn.LayerNorm(400)
        self.layernorm2 = nn.LayerNorm(nlin1)
        self.num_layers = num_layers
        self.hidden_size = nhidden
        if bidirectional:
            self.bi = 2
        else:
            self.bi = 1 
        
    def forward(self, x, lengths, maxlen, h0 = None):
        # h0 has to be shape (num_layers * num_directions, batch, hidden_size)
        if not h0:
            h0 = self.initial_state([self.num_layers*self.bi, x.data.size(0), self.hidden_size])
        r = torch.zeros(x.size(0), x.size(1), 400)
        for k in range(x.size(1)):
            r[:,k] = self.relu(self.layernorm1(self.lin_in(x[:,k,:])))
        r = pack_padded_sequence(r, lengths, batch_first = True)
        o, _ = self.gru(r, h0)
        o, lengths = pad_packed_sequence(o, batch_first = True, total_length=maxlen)
        return self.predict_proba(x, o, lengths)
    
    def initial_state(self, size):
        t = torch.FloatTensor(size[1], 100).normal_(0, 0.2)
        t = self.tanh(self.init_lin(t))
        t = t.view(size)
        return t
    
    def get_emission_params(self, h):
        h = self.relu(self.layernorm2(self.lin1(h)))
        mu = self.lin2_mu(h)
        sig = torch.exp(self.lin2_sig(h))
        sig = torch.diag(sig)
        return mu, sig
    
    def logprob_mvnormal(self, x, mu, cov):
        cov = cov + 1e-5*torch.eye(cov.size(0))
        invcov = torch.inverse(cov)
        term1 = -0.5 * torch.logdet(cov)
        term2 = torch.matmul(invcov, (x - mu))
        term3 = term1 + (-0.5 * torch.dot((x - mu),term2))
        return term3
    
    def predict_proba(self, x, hidden_seq, lengths):
        bsize = x.size(0)
        logprob = torch.zeros(bsize)
        # go over each sequence in the batch
        for i in range(bsize):
            for k in range(1, lengths[i]):
                mu,sig = self.get_emission_params(hidden_seq[i, k])
                m = multivariate_normal.MultivariateNormal(loc=mu,covariance_matrix=sig)
                # what is the probability of the current input under the model?
                logprob[i] = logprob[i].clone() +  m.log_prob(x[i,k])
                #logprob += m.log_prob(x[i, k])
                #logprob += self.logprob_mvnormal(x[i, k], mu, sig)
            logprob[i] = logprob[i].clone() / lengths[i].float()
        return logprob
from preprocessor import Preprocessor
import numpy as np
import pickle

class MarkovModel:
    '''
    t ... timestep
    z ... hidden states
    i, j... hidden state transitions
    k ... target_symbol

    '''
    def __init__(self, hidden_states, target_symbols):
        self.vocab = target_symbols
        self.voc_length = len(self.vocab)
        self.hid_length = hidden_states
        
        self.psi = {}

        self.pii = np.ones((self.hid_length)) / self.hid_length
        
        self.Aij = np.random.random((self.hid_length, self.hid_length))
        self.Aij = self.Aij / self.Aij.sum(axis=0)

        self.Bjk = np.random.random((self.hid_length, self.voc_length))
        self.Bjk = self.Bjk / self.Bjk.sum(axis=0)


    def print_progress(self, progress, text, width = 100):
        bars = '*' * int(progress * width)
        print(f'{bars: <{width}} {progress * 100:.2f}%, {text}', end='\r')
        if progress == 1:
            print('\n')


    def fwd_bckwd_probs_observing_O(self, seq):
        seq_len = len(seq)
        forward_prob = np.zeros((seq_len, self.hid_length), dtype='float32')
        backward_prob = np.zeros((seq_len, self.hid_length), dtype='float32')

        forward_prob[0] = self.pii * self.Bjk[:,seq[0]]
        backward_prob[-1] = np.ones(self.hid_length)

        for t in range(1, seq_len):
            obs_in_t = seq[t]
            obs_in_t_next = seq[-t]

            forward_prob[t] = (forward_prob[t-1] * self.Aij[:,:]) @ self.Bjk[:,obs_in_t]
            backward_prob[-t-1] = (self.Aij @ self.Bjk[:,obs_in_t_next]) * backward_prob[-t]
        
        return forward_prob, backward_prob


    def viterbi_gamma(self, seq, alpha, beta):
        seq_len = len(alpha)
        gamma = np.zeros((seq_len, self.hid_length), dtype='float32')
        xi = np.zeros((seq_len, self.hid_length, self.hid_length), dtype='float32')
        for t in range(seq_len):
            gamma[t] = (alpha[t] * beta[t]) / (alpha * beta + 1e-17).sum()

            if t == seq_len - 1: break;

            xi_nominator = (alpha[t, :] * self.Aij * self.Bjk[:, seq[t+1]] * beta[t+1,:])
            xi_denominator = xi_nominator.sum()
            xi[t] = xi_nominator / (xi_denominator +1e-17)

        return gamma, xi


    def viterbi(self, seq):
        seq_len = len(seq)
        delta = np.zeros((seq_len, self.hid_length), dtype='float32')
        psi = np.zeros((seq_len), dtype='int32')
        
        delta[0] = self.pii * self.Bjk[:, seq[0]]
        psi[0] = 0

        for t in range(1, seq_len):
            observation = seq[t]            
            delta[t] = np.max(delta[t-1] * self.Aij) * self.Bjk[:,observation]
            psi[t] = np.argmax(delta[t-1] * self.Aij)

        return np.max(delta[-1]), np.argmax(delta[-1])


    def update_params(self, gamma, xi, seq):
        seq_len = len(gamma)

        pii = gamma[0].copy()
        Aij = (xi / (1e-17+gamma.sum(axis=1)).reshape(-1,1,1)).sum(axis=0).copy()
        
        Bjk = np.zeros((self.hid_length, self.voc_length), dtype='float32')
        for k in range(self.voc_length):
            Bjk[:,k] = gamma.sum(axis=0) * (np.sum(np.array(seq) == k) / seq_len)
        
        return pii, Aij, Bjk


    def train(self, X, epochs=10):
        nsamples = len(X)
        
        for _ in range(epochs):
            alphas = []
            betas = []
            gammas = []
            xis = []
            pii = np.zeros((self.hid_length), dtype='float32')
            Aij = np.zeros((self.hid_length, self.hid_length), dtype='float32')
            Bjk = np.zeros((self.hid_length, self.voc_length), dtype='float32')
            
            cost = np.zeros(nsamples, dtype='float32')

            for n in range(nsamples):
                # FWD BCKWD
                alpha, beta = self.fwd_bckwd_probs_observing_O(X[n])
                gamma, xi  = self.viterbi_gamma(X[n], alpha, beta)
                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
                xis.append(xi)
                cost[n] = alpha[-1].sum()
                pi, A, B = self.update_params(gamma, xi, X[n])
                pii = pii + pi
                Aij = Aij + A
                Bjk = Bjk + B
                
                self.print_progress(((n+1) / nsamples), f'Cost: {cost.sum():.5f}', width = 50)
            
            print(cost.sum().round(4))

            self.pii = pii / nsamples
            self.Aij = Aij / nsamples
            self.Bjk = Bjk / nsamples




if __name__ == '__main__':
    with open('texts.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    
    p = Preprocessor()
    tokenized = p(texts)
    
    markov = MarkovModel(hidden_states = 8, target_symbols = p.vocab)
    markov.train(tokenized)
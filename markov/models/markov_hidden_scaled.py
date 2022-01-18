import numpy as np
import pickle
from utils.preprocessor import Preprocessor
import json
np.random.seed(42)

class MarkovModel:
    '''
    t ... timestep
    z ... hidden states
    i, j... hidden state transitions
    k ... target_symbol

    '''
    overall_dtype = 'float64'

    def __init__(self, hidden_states = None, vocab=None, model_path = None):
        if None is model_path:
            self.vocab = vocab
            self.voc_length = len(self.vocab)
            self.hid_length = hidden_states

            self.pii = np.ones((self.hid_length)) / self.hid_length
            
            self.Aij = np.random.random((self.hid_length, self.hid_length))
            self.Aij = (self.Aij.T / self.Aij.sum(axis=1)).T

            self.Bjk = np.random.random((self.hid_length, self.voc_length))
            self.Bjk = (self.Bjk.T / self.Bjk.sum(axis=1)).T
        else:
            self.load_model(model_path)


    def print_progress(self, progress, text, width = 100):
        bars = '*' * int(progress * width)
        print(f'{bars: <{width}} {progress * 100:.2f}%, {text}', end='\r')
        if progress == 1:
            print('\n')

    def save_model(self, fpath = 'hmmd_weights.dat'):
        with open(file = fpath, mode = 'wb') as mfile:
            model = dict(
                pii = self.pii,
                Aij = self.Aij,
                Bjk = self.Bjk
            )
            pickle.dump(model, mfile)
        with open(fpath+'_vocab.json', 'w', encoding='utf8') as vocabfile:
            vocabfile.write(json.dumps(self.vocab))
        print('model stored')

    def load_model(self, fpath):
        
        with open(fpath+'_vocab.json') as vocabfile:
            self.vocab = json.loads(vocabfile.read())

        with open(fpath, 'rb') as mfile:
            model = pickle.load(mfile)
            self.Aij = model['Aij']
            self.Bjk = model['Bjk']
            self.voc_length = len(self.vocab)
            self.hid_length = self.Aij.shape[0]
            self.pii = model['pii']
            print('model restored')

    def forward_backward(self, seq):
        seq_len = len(seq)
        alpha = np.zeros((seq_len, self.hid_length), dtype=self.overall_dtype)
        beta = np.zeros((seq_len, self.hid_length), dtype=self.overall_dtype)

        alpha[0] = self.pii * self.Bjk[:,seq[0]]
        beta[-1] = np.ones(self.hid_length)

        for t in range(1, seq_len):
            alpha[t] = (alpha[t-1] * self.Aij.T).T @ self.Bjk[:, seq[t]]
            beta[-t-1] = (beta[-t] * self.Aij.T).T @ self.Bjk[:, seq[-t]]
        
        return alpha, beta


    def viterbi_gamma(self, seq, alpha, beta):
        seq_len = len(alpha)
        gamma = np.zeros((seq_len, self.hid_length), dtype=self.overall_dtype)
        
        rseq = np.roll(seq, shift=-1)
        rbeta = np.roll(beta, shift = -1, axis=0)
        xi_nominator = (alpha[:,:,np.newaxis] * self.Aij) * (self.Bjk[:, rseq]*rbeta.T).T[:,np.newaxis,:] + self.epsilon

        xi = (xi_nominator.T / xi_nominator.sum(axis=1).sum(axis=1)[:,np.newaxis].T).T
        gamma = (alpha * beta)

        return gamma, xi


    def viterbi(self, seq):
        seq_len = len(seq)
        delta = np.zeros((seq_len, self.hid_length), dtype=self.overall_dtype)
        psi = np.zeros((seq_len), dtype='int32')
        
        delta[0] = self.pii * self.Bjk[:, seq[0]]
        psi[0] = 0

        for t in range(1, seq_len):
            observation = seq[t]            
            delta[t] = np.max(delta[t-1] @ self.Aij.T) * self.Bjk[:,observation]
            psi[t] = np.argmax(delta[t-1] @ self.Aij.T, axis=0)

        return np.max(delta[-1]).round(4), np.argmax(delta[-1]), delta, psi


    def update_params(self, pii, Aij, Bjk):
        self.pii = pii / pii.sum()
        self.Aij = (Aij / Aij.sum(axis=1)[:,np.newaxis])
        self.Bjk = (Bjk / Bjk.sum(axis=1)[:,np.newaxis])


    def calc_new_params(self, gamma, xi, seq):
        seq_len = len(gamma)

        pii = gamma[0].copy()
        Aij = np.zeros((self.hid_length, self.hid_length))
        
        for i in range(self.hid_length):
            for j in range(self.hid_length):
                Aij[i,j] += (xi[:,i,j] + self.epsilon).sum() / (gamma[:-1, i] + self.epsilon).sum()

        Bjk = np.zeros((self.hid_length, self.voc_length), dtype=self.overall_dtype)

        for t in range(seq_len):
            for k in range(self.voc_length):
                for j in range(self.hid_length):
                    Bjk[j,k] += gamma[t, j] if seq[t] == k else 0
        
        Bjk = Bjk + self.epsilon
        Bjk = (Bjk  / Bjk.sum(axis=1).T[:,np.newaxis])
        return pii, Aij, Bjk


    def train(
        self, X, 
        epochs,
        cost_thresh = 1e-05, 
        wait_epochs = 7, 
        batch_update = False,
        model_path = 'hmmd-bestmodel.dat'):

        nsamples = len(X)
        best_cost = 0.0
        wait = wait_epochs
        self.epsilon = 1e-60

        for epoch in range(epochs):
            alphas = []
            betas = []
            gammas = []

            pii = np.zeros((self.hid_length), dtype=self.overall_dtype)
            Aij = np.zeros((self.hid_length, self.hid_length), dtype=self.overall_dtype)
            Bjk = np.zeros((self.hid_length, self.voc_length), dtype=self.overall_dtype)
            
            cost = np.zeros(nsamples, dtype=self.overall_dtype)

            for n in range(nsamples):
                sample = X[n]
                # FWD BCKWD
                alpha, beta = self.forward_backward(sample)
                c = np.expand_dims(alpha.sum(axis=1) + self.epsilon, axis=1)
                alpha, beta = alpha/c, beta/c
                #beta[-1] = np.ones((self.hid_length))

                gamma, xi = self.viterbi_gamma(sample, alpha, beta)
                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
                
                cost[n] = np.product(alpha, axis=0).sum()
                pi, A, B = self.calc_new_params(gamma, xi, sample)
                
    
                if batch_update:
                    pii = pii + pi
                    Aij = Aij + A
                    Bjk = Bjk + B
                else:
                    self.update_params(pi, A, B)
                
                self.print_progress(((n+1) / nsamples), f'Epoch {epoch} / Forward Prob: {cost.sum():.4f}', width = 50)
            
            if batch_update:
                self.update_params(pii, Aij, Bjk)

            if cost.sum() - best_cost < cost_thresh:
                print(f'No improvement. Waiting {wait} further epochs')
                if wait == 0:
                    break;
                wait -=1
            else:
                best_cost = cost.sum()
                self.save_model(model_path)
                wait = wait_epochs
            
        

if __name__ == '__main__':
    #with open('examples/texts.txt','r',encoding='utf8') as f:
    #    texts = [line for line in f.read().split('\n') if len(line.strip())]
    
    with open('examples/coin_flip.txt','r',encoding='utf8') as f:
        texts = [' '.join(list(line.strip())) for line in f]
    
    p = Preprocessor()
    tokenized = p(texts)
    len(tokenized)

    markov = MarkovModel(hidden_states = 2, vocab = p.w2id)
    markov.train(tokenized, epochs = 100, batch_update = True, model_path='hmmd-scaled.dat')
    print(p.vocab)
    print(markov.vocab)
    print('A', markov.Aij.round(2))
    print('B', markov.Bjk.round(2))
import numpy as np
import pickle
from utils.preprocessor import Preprocessor
import json


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
            alpha[t] /= alpha[t].sum()
            beta[-t-1] = (beta[-t] * self.Aij.T).T @ self.Bjk[:, seq[-t]]
            beta[-t-1] /= beta[-t-1].sum()
        
        return alpha, beta


    def viterbi_gamma(self, seq, alpha, beta):
        seq_len = len(alpha)
        gamma = np.zeros((seq_len, self.hid_length), dtype=self.overall_dtype)
        
        rseq = np.roll(seq, shift=-1)
        rbeta = np.roll(beta, shift = -1, axis=0)
        xi_nominator = (alpha[:,:,np.newaxis] * self.Aij) * (self.Bjk[:, rseq]*rbeta.T).T[:,np.newaxis,:]
        xi = (xi_nominator.T / (xi_nominator.sum(axis=1).sum(axis=1)[:,np.newaxis].T + self.epsilon)).T
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

        Aij = Aij * self.lr + self.Aij
        Bjk = Bjk * self.lr + self.Bjk

        self.Aij = (Aij.T / (Aij.sum(axis=1) + self.epsilon)).T
        self.Bjk = (Bjk.T / (Bjk.sum(axis=1) + self.epsilon)).T


    def calc_new_params(self, gamma, xi, seq):
        pii = gamma[0].copy()
        Aij = (xi[:-1].sum(0).T / (gamma[:-1,:].sum(0) + self.epsilon)).T
        Bjk = (gamma.T @ self.eye[seq])
        Bjk = Bjk  / (Bjk.sum(axis=1).T[:,np.newaxis])
        return pii, Aij, Bjk


    def train(
        self, X, 
        epochs,
        cost_thresh = 1e-05, 
        wait_epochs = 7, 
        batch_update = False,
        lr = 0.1,
        randomize = 0,
        model_path = 'hmmd-bestmodel.dat'):

        nsamples = len(X)
        best_cost = 0.0
        wait = wait_epochs
        self.epsilon = 0.0
        self.lr = lr
        self.randomize = randomize

        self.eye = np.eye(self.voc_length)

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
                seq_len = len(sample)
                # FWD BCKWD
                alpha, beta = self.forward_backward(sample)
                c = np.expand_dims(alpha.sum(axis=1) + self.epsilon, axis=1)
                alpha, beta = alpha, beta
                #beta[-1] = np.ones((self.hid_length))

                gamma, xi = self.viterbi_gamma(sample, alpha, beta)
                alphas.append(alpha)
                betas.append(beta)
                gammas.append(gamma)
                
                cost[n] = np.product(alpha, axis=0).sum()
                pi, A, B = self.calc_new_params(gamma, xi, sample)
                
    
                if batch_update:
                    pii = pii + (pi / nsamples)
                    Aij = Aij + (A / nsamples)
                    Bjk = Bjk + (B / nsamples)
                else:
                    self.update_params(pi, A, B)
                
                self.print_progress(((n+1) / nsamples), f'Epoch {epoch} / Forward Prob: {cost.sum():.4f}', width = 50)
            
            if batch_update:
                self.update_params(pii, Aij, Bjk)

            if cost.sum() - best_cost <= cost_thresh:
                print(f'No improvement. Waiting {wait} further epochs')
                
                if int(wait_epochs - wait) in list(range(int(wait_epochs/2))):
                    # add some randomness, if no improvement can be found... 
                    # just to make sure... modifiy two numbers
                    for _ in range(min(self.randomize, self.hid_length**2)):
                        p1, p2 = np.random.randint(0, self.hid_length, size=2)
                        self.Aij[p1, p2] += np.clip(np.random.random(1)*2 - 1.0, a_min=0.0, a_max=1.0)
                        self.update_params(self.pii, self.Aij, self.Bjk)
                if wait == 0:
                    break;
                wait -=1
            else:
                best_cost = cost.sum()
                self.save_model(model_path)
                wait = wait_epochs
            
        

if __name__ == '__main__':
    with open('examples/texts.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    np.random.seed(42)

    #with open('examples/coin_flip.txt','r',encoding='utf8') as f:
    #    texts = [' '.join(list(line.strip())) for line in f]
    
    p = Preprocessor()
    tokenized = p(texts)
    len(tokenized)

    markov = MarkovModel(hidden_states = 8, vocab = p.w2id)
    markov.train(
        tokenized, 
        epochs = 100, 
        batch_update = True, 
        lr = 0.3,
        wait_epochs = 30,
        cost_thresh = 1e-05,
        randomize = 30,
        model_path='hmmd-scaled.dat')

    print(p.vocab)
    print(markov.vocab)
    print('pii', markov.pii.round(2))
    print('A', markov.Aij.round(2))
    print('B', markov.Bjk.round(2))
from turtle import forward
import numpy as np
from utils.preprocessor import Preprocessor
from scipy.stats import multivariate_normal
from utils.tools import scale
import wave

class HMMGMM:
    def __init__(self, hidden_states, gaussians):
        self.nhidden_states = hidden_states
        self.ngaussians = gaussians
        self.init_init_states()
        self.init_hidden_states()

    def init_init_states(self):
        self.pii = np.random.random(size = self.nhidden_states)
        self.pii = scale(self.pii)
  
    def init_hidden_states(self):
        self.Aij = np.random.random(size = (self.nhidden_states, self.nhidden_states))
        self.Aij = scale(self.Aij)
    
    def build(self, X):
        self.output_dims = X.shape[2]
        self.init_gaussians(X)

    def init_gaussians(self, sequence):
        self.weights_for_gaussians = np.random.random(size = (self.nhidden_states, self.ngaussians))
        self.weights_for_gaussians = scale(self.weights_for_gaussians)
        
        self.gauss_means = np.zeros((self.nhidden_states, self.ngaussians, self.output_dims))
        for dim in range(self.output_dims):
            self.gauss_means[...,dim] = np.random.choice(sequence[...,dim].reshape(-1), size=(self.nhidden_states, self.ngaussians))
        
        eye = np.eye(self.output_dims)
        self.covs = np.array([[[eye[ii] * 5.0 for ii in range(self.output_dims)] for _ in range(self.ngaussians)] for _ in range(self.nhidden_states)])
        


    def calc_gauss_res_per_step(self, x):
        ''' This is the very heart of the process'''
        gauss_result_per_step = np.zeros((len(x), self.nhidden_states, self.ngaussians))
        for state in range(self.nhidden_states):
            for k in range(self.ngaussians):
                mu, c = self.gauss_means[state,k], self.covs[state, k]
                gauss_result_per_step[:, state, k] = multivariate_normal.pdf(x, mean = mu, cov = c)
        return gauss_result_per_step


    def forward_backward(self, seq):
        seqlen = len(seq)
        alpha, beta = np.zeros((seqlen, self.nhidden_states)), np.zeros((seqlen, self.nhidden_states))

        components = self.calc_gauss_res_per_step(seq) * self.weights_for_gaussians
        B = components.sum(axis = -1)
        
        alpha[0] = self.pii @ B[0]
        alpha[0] = scale(alpha[0])

        beta[-1] = np.ones(self.nhidden_states)

        for t in range(1, seqlen):
            alpha[t] = alpha[t-1]@self.Aij * B[t]
            #alpha[t] = scale(alpha[t])
            beta[-t-1] = self.Aij@(B[-t] * beta[-t])
            #beta[-t-1] = scale(beta[-t-1])
        
        P = alpha[-1].sum()
        
        # update for Gaussians
        factors = scale(alpha * beta, on_axis=0) # old gamma
        gamma = (components * factors[...,np.newaxis]) / B[..., np.newaxis]

        return alpha, beta, gamma, B, P


    def calc_new_params(self, alpha, beta):
         pi = alpha[0] * beta[0] / alpha[-1].sum()



    def train(self, X, epochs = 0):
        self.build(X)
        for epoch in range(epochs):
            for seq in X:
                alphas, betas, gammas = [], [], []
                Bs = []
                Ps = []
                alpha, beta, gamma, B , P = self.forward_backward(seq)
                alphas.append(alpha); betas.append(beta); gammas.append(gamma)
                Bs.append(B); Ps.append(P)

                pii, Aij, gweights, means, covs = self.calc_new_params(alpha, beta)



def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    np.random_state=42
    #spf = wave.open('examples/hmm_class_helloworld.wav', 'r')
    #signal_in = spf.readframes(-1)
    #signal = np.fromstring(signal_in, 'int16')
    #T = len(signal)

    #double_signal = np.stack([signal, signal * 0.5], axis=1)

    double_signal = np.stack([
        np.random.normal(10, 10, 20), 
        np.random.normal(-1, 2, 20),
        np.random.normal(3, 4, 20)], axis=1).astype('int16')
    
    hmgmm = HMMGMM(
        hidden_states = 10,
        gaussians = 3)

    hmgmm.train(double_signal[np.newaxis,...], epochs = 10)

        

if __name__ == '__main__':
    main()
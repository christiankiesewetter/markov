from turtle import forward
import numpy as np
from utils.preprocessor import Preprocessor
from scipy.stats import multivariate_normal
from utils.tools import scale
import wave

class HMMGMM:
    def __init__(self, hidden_states, gaussians, lr):
        self.lr = lr
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
        self.ndims_out = X.shape[2]
        self.init_gaussians(X)

    def init_gaussians(self, sequence):
        self.weights_for_gaussians = np.random.random(size = (self.nhidden_states, self.ngaussians))
        self.weights_for_gaussians = scale(self.weights_for_gaussians)
        
        self.gauss_means = np.zeros((self.nhidden_states, self.ngaussians, self.ndims_out))
        for dim in range(self.ndims_out):
            self.gauss_means[...,dim] = np.random.choice(sequence[...,dim].reshape(-1), size=(self.nhidden_states, self.ngaussians))
        
        eye = np.eye(self.ndims_out)
        self.covs = np.array([[[eye[ii] * 5.0 for ii in range(self.ndims_out)] for _ in range(self.ngaussians)] for _ in range(self.nhidden_states)])
        


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
        B = components.sum(axis = 2)
        
        alpha[0] = self.pii * B[0]
        alpha[0] = scale(alpha[0])
        beta[-1] = np.ones(self.nhidden_states)

        for t in range(1, seqlen):
            alpha[t] = alpha[t-1]@self.Aij * B[t]; alpha[t] = scale(alpha[t])
            beta[-t-1] = self.Aij@(B[-t] * beta[-t]); beta[-t-1] = scale(beta[-t-1])
            
        
        P = alpha[-1].sum()
        
        # update for Gaussians
        factors = scale(alpha * beta, on_axis = 0) # old gamma, scaled
        gamma = (components * factors[...,np.newaxis]) / B[..., np.newaxis]

        return alpha, beta, gamma, P, B


    def calc_new_params(self, alpha, beta, gamma, B, seq):
        pii = alpha[0] * beta[0] 
        gweights = (gamma.T / gamma.sum(axis=0).sum(axis=1)[np.newaxis,:,np.newaxis]).T.sum(axis=0) # ???? Check again
        means = (gamma[:,:,:,np.newaxis] * seq[:, np.newaxis,np.newaxis,:]).sum(axis=0) / gamma.sum(axis=0)[:,:,np.newaxis]

        covs = np.zeros((self.nhidden_states, self.ngaussians, self.ndims_out, self.ndims_out))
        for state in range(self.nhidden_states):
            for k in range(self.ngaussians):
                for t in range(len(seq)):
                    covs[state, k] += gamma[t, state, k] * ((seq - self.gauss_means[state, k]).T@(seq - self.gauss_means[state, k]))
                covs[state, k] = covs[state, k] / gamma[:, state, k].sum()
        
        
        a_den_n = (alpha[:-1] * beta[:-1]).sum(axis=0, keepdims=True).T 
        
        # numerator for A
        a_num_n = np.zeros((self.nhidden_states, self.nhidden_states))
        for i in range(self.nhidden_states):
            for j in range(self.nhidden_states):
                for t in range(len(seq)-1):
                    a_num_n[i,j] += alpha[t,i] * self.Aij[i,j] * B[t+1,j] * beta[t+1,j]
        
        Aij = a_num_n / a_den_n

        return pii, Aij, gweights, means, covs


    def update_params(self, epoch_pii, epoch_Aij, epoch_gweights, epoch_means, epoch_covs):
        self.pii = scale(epoch_pii)
        self.Aij = scale(epoch_Aij)
        self.weights_for_gaussians = scale(epoch_gweights)
        self.gauss_means = epoch_means
        self.covs = epoch_covs


    def train(self, X, epochs = 3):
        self.build(X)
        for epoch in range(epochs):
            epoch_pii = np.zeros_like(self.pii)
            epoch_Aij = np.zeros_like(self.Aij)
            epoch_gweights = np.zeros_like(self.weights_for_gaussians)
            epoch_means = np.zeros_like(self.gauss_means)
            epoch_covs = np.zeros_like(self.covs)
            for ii in range(len(X)):
                seq = X[ii]
                alpha, beta, gamma, P, B = self.forward_backward(seq)
                pii, Aij, gweights, means, covs = self.calc_new_params(alpha, beta, gamma, B, seq)
                
                epoch_pii += pii / (epochs * P)
                epoch_Aij += Aij / (epochs * P)
                epoch_gweights += gweights / (epochs * P)
                epoch_means += means / (epochs * P)
                epoch_covs += covs / (epochs * P)
            
            self.update_params(epoch_pii, epoch_Aij, epoch_gweights, epoch_means, epoch_covs)
            print(epoch)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    np.random_state=42
  
    double_signal = np.stack([
        np.random.normal(10, 10, 20), 
        np.random.normal(-1, 2, 20),
        np.random.normal(3, 4, 20)], axis=1).astype('float32')
    
    double_signal = (double_signal - double_signal.min(axis=0))
    
    hmgmm = HMMGMM(
        hidden_states = 10,
        gaussians = 4,
        lr = 0.1)

    hmgmm.train(double_signal[np.newaxis,...], epochs = 10)

        

if __name__ == '__main__':
    main()
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def gmm(X, kclusters, max_iter=20, smoothing=1e-2):
    samples, dimensions = X.shape
    means = np.zeros((kclusters, dimensions))
    clusterprobabilities = np.zeros((samples, kclusters))
    covariances = np.zeros((kclusters, dimensions, dimensions))
    pi = np.ones(kclusters) / kclusters # uniform probability the point comes from cluster k

    for k in range(kclusters):
        means[k] = X[np.random.choice(samples)]
        covariances[k] = np.eye(dimensions)

    lls = []
    weighted_pdfs = np.zeros((samples, kclusters)) # we'll use these to store the PDF value of sample n and Gaussian k
    for i in range(max_iter):
        for k in range(kclusters):
            weighted_pdfs[:,k] = pi[k] * multivariate_normal.pdf(X, means[k], covariances[k])

        clusterprobabilities = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

        for k in range(kclusters):
            Nk = clusterprobabilities[:,k].sum()
            pi[k] = Nk / samples
            means[k] = clusterprobabilities[:,k].dot(X) / Nk

            delta = X - means[k] # samples x dimensions
            Rdelta = np.expand_dims(clusterprobabilities[:,k], -1) * delta # multiplies R[:,k] by each col. of delta - samples x dimensions
            covariances[k] = Rdelta.T.dot(delta) / Nk + np.eye(dimensions)*smoothing # dimensions x dimensions

        ll = np.log(weighted_pdfs.sum(axis=1)).sum()
        lls.append(ll)
        if i > 0:
            if np.abs(lls[i] - lls[i-1]) < 0.1:
                break

    plt.plot(lls)
    plt.title("Log-Likelihood")
    plt.show()

    random_colors = np.random.random((kclusters, 3))
    colors = clusterprobabilities.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

    print("pi:", pi)
    print("means:", means)
    print("covariances:", covariances)
    return clusterprobabilities


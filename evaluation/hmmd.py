from markov.models.markov_hidden_scaled import (
    MarkovModel as HiddenMarkov, 
)
import numpy as np

if __name__ == '__main__':
    hmm = HiddenMarkov(model_path='hmmd-scaled.dat')
    sequence = "Ja ja sagte er freudig"
    sequence = [hmm.vocab[w.lower()] for w in sequence.split()]
    id2w = {w:k for k, w in hmm.vocab.items()}
    print(' '.join([id2w[w] for w in sequence]))
    print(sequence)
    perc, state, delta, psi = hmm.viterbi(sequence)
    print(np.argmax(delta, axis=1))
    print(psi)
    print(perc)
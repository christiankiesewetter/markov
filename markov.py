import numpy as np
from itertools import product
import re

unchar = re.compile(r'[,.\-;:_#\'+*?!"&%$§"()]')

class MarkovModel:
    config = dict()
    __vocab = dict()
    A_ij = dict()
    __overall_length = 0
    vocab = dict()
    
    def __init__(self, **kwargs):
        self.config.update(kwargs)
    
    def calc_freqs_for_word(self, word, seq, order):
        picks = np.array([False for _ in range(len(seq))])
        for pos in range(0, len(seq)):
            if ' '.join(seq[pos:pos+order]) == ' '.join(word) and (pos+order < len(picks)):
                picks[pos+order] = True
        
        words, counts = np.unique(seq[picks], return_counts=True)
        words = np.array([self.vocab['word'][w] for w in words])
        return words, counts

    def setup_n_order_vocab(self, sequences, order):
        words = [m for m in product(sequences, repeat = order)]
        id2w = dict(enumerate(words))
        w2id = {w:id for id, w in id2w.items()}
        self.__vocab.update({
            f'id2w{order}' : id2w,
            f'w2id{order}' : w2id
        })

    def setup_vocab(self, allsequences):
        words, counts = np.unique(allsequences, return_counts=True)
        self.__overall_length = counts.sum()
        self.__rel_probs = counts/self.__overall_length
        self.vocab['id'] = dict(enumerate(words))
        self.vocab['word'] = {v:k for k,v in self.vocab['id'].items()}

    def train_frequency_probabilites(self, epsilon):
        inputs = len(self.vocab['id'])
        self.pi_i = np.ones_like((inputs)) * epsilon + self.__rel_probs

    def train_n_order(self, allsequences, order, epsilon):
        inputs1 = len(self.__vocab[f'id2w{order}'])
        outputs = len(self.vocab['id'])
        A_ij = np.ones((inputs1, outputs)) * epsilon
        for w, id in self.__vocab[f'w2id{order}'].items():
            wm, cm = self.calc_freqs_for_word(w, allsequences, order)
            a_j = np.zeros(outputs)
            if len(wm != 0):
                np.put(a_j, wm, cm)
            A_ij[id] = a_j

        self.A_ij.update({order: np.log(A_ij + epsilon) / outputs})


    def train(self, sequences, order, epsilon = 1e-10):
        allsequences = np.array([unchar.sub(r'', word).lower() for line in sequences for word in line.split() if len(line.strip())>0 and len(word.strip())>0])

        self.setup_vocab(allsequences)
        self.train_frequency_probabilites(epsilon)
        
        self.setup_n_order_vocab(sequences = allsequences, order = order)
        self.train_n_order(allsequences, order = order, epsilon = epsilon)

    def __len__(self):
        return self.__overall_length
    

    def __call__(self, begin, order, nsteps = 10):
        seq = begin.lower().split(' ')
        in_id = self.__vocab[f'w2id{order}'][tuple(seq[:order])]
        for step in range(nsteps):
            out_id = np.argmax(self.A_ij[order][in_id])
            seq.append(self.vocab['id'][out_id])
            in_id = self.__vocab[f'w2id{order}'][tuple(seq[step+1:step+order+1])]
        return ' '.join(seq)



if __name__ == '__main__':
    with open('texts2.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    mm = MarkovModel()
    mm.train(texts, order = 2, epsilon = 0)
    print(mm('es waren', order = 2, nsteps = 30))
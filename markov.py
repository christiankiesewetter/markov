import numpy as np
from itertools import product
import re

unchar = re.compile(r'[,.\-;:_#\'+*?!"&%$§"()]')

class MarkovModel:
    config = dict()
    __vocab = dict.fromkeys(['w2id','id2w','id2freq'])
    A_ij = dict()
    __overall_length = 0
    
    def __init__(self, **kwargs):
        self.config.update(kwargs)
    
    def encode_seq(self, seq):
        return np.array([self.__vocab['w2id'][word] for word in seq])

    def calc_freq_for_word(self, word, sequence):
        adjacent_words = np.roll(sequence == word, 1)
        words, counts = np.unique(sequence[adjacent_words], return_counts=True)
        return words, counts

    def setup_n_order_vocab(self, words, order):
        id2w = dict(enumerate(words))
        w2id = {w:id for id,w in id2w.items()}
        self.__vocab.update({
            f'id2w{order}':id2w,
            f'w2id{order}':w2id,
        })

    def setup_vocab(self, allsequences):
        words, counts = np.unique(allsequences, return_counts=True)
        self.__overall_length = counts.sum()
        self.__rel_probs = counts/self.__overall_length
        self.setup_n_order_vocab(words = words, order = 1)
        self.setup_n_order_vocab(words = [' '.join(m) for m in product(allsequences, repeat=2)], order = 2)

    def train_frequency_probabilites(self, epsilon):
        inputs = len(self.__vocab['id2w1'])
        self.pi_i = np.ones_like((inputs)) * epsilon + self.__rel_probs

    def train_n_order(self, allsequences, order, epsilon):
        inputs1 = len(self.__vocab[f'id2w{order}'])
        outputs = len(self.__vocab['id2w'])
        A_ij = np.ones((inputs1, outputs)) * epsilon
        for _, id in self.__vocab[f'w2id{order}'].items():
            wm, cm = self.calc_freq_for_word(id, self.encode_seq(allsequences))
            a_j = np.zeros(outputs)
            np.put(a_j, wm, cm)
            A_ij[id] = a_j

        self.A_ij.update({order: np.log(A_ij + epsilon) / outputs})

    def train(self, sequences, epsilon = 1e-10):
        allsequences = np.array([unchar.sub(r'', word).lower() for line in sequences for word in line.split() if len(line.strip())>0 and len(word.strip())>0])

        self.setup_vocab(allsequences)
        self.train_frequency_probabilites(epsilon)
        self.train_n_order(allsequences, order = 1, epsilon = epsilon)
        self.train_n_order(allsequences, order = 2, epsilon = epsilon)

    def __len__(self):
        return self.__overall_length

    def __call__(self, begin, nsteps = 10):
        seq = [begin]
        next_id = self.encode_seq([begin])
        for step in range(nsteps):
            next_id = np.argmax(self.A_ij[next_id])
            seq.append(self.__vocab['id2w'][next_id])
        return ' '.join(seq)



if __name__ == '__main__':
    with open('texts2.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    mm = MarkovModel()
    mm.train(texts)
    
    #print(mm('ist', nsteps=30))
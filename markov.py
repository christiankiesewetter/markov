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
    
    def calc_freqs_for_word(self, word, seq, order, epsilon):
        picks = np.array([False for _ in range(len(seq))])
        for pos in range(0, len(seq)):
            if ' '.join(seq[pos:pos+order]) == ' '.join(word) and (pos+order < len(picks)):
                picks[pos+order] = True
        
        words, counts = np.unique(seq[picks], return_counts=True)
        words = np.array([self.vocab['word'][w] for w in words])
        freq = np.log(counts  + epsilon / counts.sum())
        return dict(zip(words, freq))

    def setup_n_order_vocab(self, sequences, order):
        words = set((' '.join(sequences[pos:pos+order]) for pos in range(len(sequences) - order)))
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

    def print_progress(self, progress, order, width = 100):
        if progress == 1.0:
            print('\n')
        else:
            bars = '*' * int(progress * width)
            print(f'{bars: <{width}} {progress * 100:.2f}%, order {order}', end='\r')

    def train_n_order(self, sequence, order, epsilon):
        self.A_ij.update({order:dict()})
        inputs = len(self.__vocab[f'w2id{order}'])
        for num, (w, id) in enumerate(self.__vocab[f'w2id{order}'].items()):
            sparse_dict = self.calc_freqs_for_word(w, sequence, order, epsilon)
            if len(sparse_dict) > 0:
                self.A_ij[order].update({id : sparse_dict})
            self.print_progress(num / inputs, order)

    def train(self, sequences, orders, epsilon = 1e-10):
        allsequences = []
        for line in sequences: 
            for word in line.split():
                if len(line.strip())>0 and len(word.strip())>0:
                    text = unchar.sub(r'', word).lower()
                    if text.strip() != '':
                        allsequences.append(text)
        
        allsequences = np.array(allsequences)

        self.setup_vocab(allsequences)
        self.train_frequency_probabilites(epsilon)
        
        for order in orders:
            self.setup_n_order_vocab(sequences = allsequences, order = order)
            self.train_n_order(allsequences, order = order, epsilon = epsilon)

    def __len__(self):
        return self.__overall_length
    

    def __call__(self, begin, order, nsteps = 10):
        seq = begin.lower().split(' ')
        in_id = self.__vocab[f'w2id{order}'][tuple(seq[:order])]
        for step in range(nsteps):
            out_id = sorted(self.A_ij[order][in_id].items(), key = lambda m: m[1])[0][0]
            seq.append(self.vocab['id'][out_id])
            in_id = self.__vocab[f'w2id{order}'][tuple(seq[step+1:step+order+1])]
        return ' '.join(seq)



if __name__ == '__main__':
    with open('texts.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    mm = MarkovModel()
    mm.train(texts, orders = [1,2])
    print(mm('es waren', order = 2, nsteps = 30))
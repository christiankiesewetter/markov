import os
import re
import pickle
import random
from itertools import product
import numpy as np

unchar = re.compile(r'[,.\-;:_#\'+*?!"&%$§"()]')

class MarkovModel:
    __vocab = dict()
    A_ij = dict()
    vocab = dict()
    __overall_length = 0
    epsilon = 1e-04
    
    def __init__(self, model_path = None, epsilon = None):
        if not None is model_path and os.path.isfile(model_path):
            with open(model_path, 'rb') as mfile:
                model = pickle.load(mfile)
                self.A_ij = model['Aij']
                self.__vocab = model['internal_vocab'] 
                self.vocab = model['external_vocab']
                print('model loaded')

        if not None is epsilon:
            self.epsilon = epsilon


    def save(self, model_path = None):
        with open(file = f'{model_path}', mode = 'wb') as mfile:
            model = dict(
                Aij = self.A_ij,
                internal_vocab = self.__vocab,
                external_vocab = self.vocab
            )
            pickle.dump(model, mfile)
            print('model stored')
        
    def calc_freqs_for_word(self, word, seq, order):
        picks = np.array([False for _ in range(len(seq))])
        for pos in range(0, len(seq)-order):
            if ' '.join(seq[pos:pos+order]) == word:
                picks[pos+order] = True
        
        words, counts = np.unique(seq[picks], return_counts=True)
        words = np.array([self.vocab['word'][w] for w in words])
        freq = np.log(counts  + self.epsilon / counts.sum())
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

    def train_frequency_probabilites(self):
        inputs = len(self.vocab['id'])
        self.pi_i = np.ones_like((inputs)) * self.epsilon + self.__rel_probs

    def print_progress(self, progress, order, width = 100):
        bars = '*' * int(progress * width)
        print(f'{bars: <{width}} {progress * 100:.2f}%, order {order}', end='\r')
        if progress == 1:
            print('\n')


    def train_n_order(self, sequence, order):
        self.A_ij.update({order:dict()})
        inputs = len(self.__vocab[f'w2id{order}'])
        for num, (w, id) in enumerate(self.__vocab[f'w2id{order}'].items()):
            sparse_dict = self.calc_freqs_for_word(w, sequence, order)
            if len(sparse_dict) > 0:
                self.A_ij[order].update({id : sparse_dict})
            self.print_progress((num + 1) / inputs, order)

    def train(self, sequences, orders):
        allsequences = []
        for line in sequences: 
            for word in line.split():
                if len(line.strip())>0 and len(word.strip())>0:
                    text = unchar.sub(r'', word)
                    if text.strip() != '':
                        allsequences.append(text.lower().strip())
        
        allsequences = np.array(allsequences)

        self.setup_vocab(allsequences)
        self.train_frequency_probabilites()
        
        for order in orders:
            self.setup_n_order_vocab(sequences = allsequences, order = order)
            self.train_n_order(allsequences, order = order)

    def __len__(self):
        return self.__overall_length
    
    def next_id(self, order, wkey):
        curr_order = order
        curr_wkey = wkey

        while not curr_wkey in self.__vocab[f'w2id{curr_order}'] and curr_order > 0:
            curr_order -= 1
            curr_wkey = ' '.join(curr_wkey.split(' ')[-curr_order:])

        if curr_order > 0:
            in_id = self.__vocab[f'w2id{curr_order}'][curr_wkey]
        else:
            in_id = int(random.random() * len(self.__vocab[f'w2id1']))
        
        return in_id, curr_order
    
    def sample_word(self,order, in_id):
        p = random.random()
        if p <= self.epsilon: # Case, when very low values are accepted, we randomly choose something
            out_word = random.choice(list(self.vocab['id'].values()))
            return out_word

        sortedwords = sorted(self.A_ij[order][in_id].items(), key = lambda m: m[1], reverse = True)
        out_id = sortedwords[-1][0] # default value
        for (iid, prob) in sortedwords:
            if prob >= p:
                out_id = iid
                break
        return self.vocab['id'][out_id]

    def __call__(self, begin, order, nsteps = 10):
        seq = begin.lower().split(' ')
        wkey = ' '.join(seq[-order:])
        in_id, curr_order = self.next_id(order, wkey)
        
        for step in range(nsteps):
            next_word = self.sample_word(curr_order, in_id)
            seq.append(next_word)
            wkey = ' '.join(seq[-order:])
            in_id, curr_order = self.next_id(order, wkey)

        return ' '.join(seq)


if __name__ == '__main__':
    with open('texts.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    
    mm = MarkovModel('mm1.dat', epsilon = 1e-02)
    #mm.train(texts, orders = [1,2,3,4])
    #mm.save('mm1.dat')

    print(mm('Warum', order = 4, nsteps = 20))
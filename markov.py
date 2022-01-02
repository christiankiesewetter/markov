import os
import re
import pickle
import random
from itertools import product
import numpy as np
from collections import namedtuple

unchar = re.compile(r'[,.\-;:_#\'+*?!"&%$§"()]')

ProbSeq = namedtuple('ProbSequence', 'step prob order')

class MarkovModel:
    __vocab = dict()
    pi_i = dict()
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
                self.pi_i = model['pi_i']
                print('model loaded')

        if not None is epsilon:
            self.epsilon = epsilon


    def save(self, model_path = None):
        with open(file = f'{model_path}', mode = 'wb') as mfile:
            model = dict(
                Aij = self.A_ij,
                internal_vocab = self.__vocab,
                external_vocab = self.vocab,
                pi_i = self.pi_i
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
        freq = counts / (counts.sum() + self.epsilon)
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
        intermediate_pi_i = dict()
        actual_lines = 0
        for line in sequences: 
            splitline = line.split()
            for word in splitline:
                if len(line.strip())>0 and len(word.strip())>0:
                    isfirstword = True
                    text = unchar.sub(r'', word)
                    if text.strip() != '':
                        entry = text.lower().strip()
                        allsequences.append(entry)
                        if isfirstword:
                            intermediate_pi_i[entry] = intermediate_pi_i.get(entry, 0) + 1
                            actual_lines += 1
                            isfirstword = False

        self.pi_i = {k:(v/actual_lines) for k, v in intermediate_pi_i.items()}
        allsequences = np.array(allsequences)

        self.setup_vocab(allsequences)
        
        for order in orders:
            self.setup_n_order_vocab(sequences = allsequences, order = order)
            self.train_n_order(allsequences, order = order)

    def __len__(self):
        return self.__overall_length
    
    def next_id(self, order, wkey):
        curr_order = order
        curr_wkey = wkey

        while curr_order > 0 and not curr_wkey in self.__vocab[f'w2id{curr_order}']:
            curr_order -= 1
            curr_wkey = ' '.join(curr_wkey.split(' ')[-curr_order:])

        if curr_order > 0:
            in_id = self.__vocab[f'w2id{curr_order}'][curr_wkey]
        else:
            in_id = self.process_sample([(self.vocab['word'][k], v) for k,v in self.pi_i.items()])
        
        return in_id, curr_order
    

    def process_sample(self, sampleitems):
        p = random.random()
        if p <= self.epsilon: # Case, when very low values are accepted, we randomly choose something
            out_word = random.choice(list(self.vocab['id'].values()))
            return out_word
        
        sortedwords = sorted(sampleitems, key = lambda m: np.log(m[1]), reverse = True)
        out_id = sortedwords[-1][0] # default value
        for (iid, prob) in sortedwords:
            if prob >= p:
                out_id = iid
                break
        return out_id


    def sample_word(self, order, in_id):
        out_id = self.process_sample(self.A_ij[order][in_id].items())
        return self.vocab['id'][out_id]


    def generate(self, begin, order, nsteps = 10):
        seq = begin.lower().split(' ')
        wkey = ' '.join(seq[-order:])
        in_id, curr_order = self.next_id(order, wkey)
        if curr_order == 0:
            curr_order = 1
            seq[0] = self.vocab['id'][in_id]

        for step in range(nsteps):
            next_word = self.sample_word(curr_order, in_id)
            seq.append(next_word)
            wkey = ' '.join(seq[-order:])
            in_id, curr_order = self.next_id(order, wkey)

        return ' '.join(seq)
    

    def __call__(self, sequence, order):
        '''
        Calculates the probability of the occuring sequence.
        '''
        seq = sequence.lower().split(' ')
        prob_sequence = [ProbSeq(0, self.pi_i[seq[0].lower().strip()], 0)] # initialize
        for step in range(1, len(seq)):
            curr_order = min(step, order)
            w_i = ' '.join(seq[step-curr_order:step])
            w_j = seq[step]

            order_dep_id_w_i = self.__vocab[f'w2id{curr_order}'][w_i]
            order_dep_id_w_j = self.vocab['word'][w_j]

            if order_dep_id_w_i in self.A_ij[curr_order] \
                and order_dep_id_w_j in self.A_ij[curr_order][order_dep_id_w_i]:
                    prob = self.A_ij[curr_order][order_dep_id_w_i][order_dep_id_w_j]
            else:
                prob = self.epsilon

            prob_sequence.append(ProbSeq(step, prob, curr_order))
        
        return np.prod([a.prob for a in prob_sequence]), prob_sequence



if __name__ == '__main__':
    with open('texts.txt','r',encoding='utf8') as f:
        texts = [line for line in f.read().split('\n') if len(line.strip())]
    
    mm = MarkovModel('mm1.dat', epsilon = 1e-03)
    #mm.train(texts, orders = [1,2,3])
    #mm.save('mm1.dat')

    print(mm.generate('Du', order = 3, nsteps = 50))
    print(mm('es waren zweiundzwanzig', order = 2)[0])
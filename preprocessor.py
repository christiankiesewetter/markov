import re

unchar = re.compile(r'[,.\-;:_#\'+*?!"&%$§"()]')

class Preprocessor:
    def __init__(self):
        self.w2id = dict()
        self.id2w = dict()
        self.initial_frequencies = dict()
        self.frequencies = dict()


    def clean_sequences(self, sequences):
        allsequences = []
        for line in sequences:
            truncated_line = line.strip()
            if len(truncated_line)>0:
                splitline = unchar.sub(r'', truncated_line.lower()).split()
                allsequences.append(splitline)
        return allsequences


    def setup_vocab(self, sequences):
        words = set('<UNK>')
        count = 0
        for sequence in sequences :
            for pos in range(len(sequence)):
                entry = ' '.join(sequence[pos:pos+1])
                words.add(entry)
                if pos == 0:
                    self.initial_frequencies.update({entry: self.initial_frequencies.get(entry, 0) + 1 })
                self.frequencies.update({entry: self.frequencies.get(entry, 0) + (1 / len(sequences))})
                count += 1
        
        self.vocab = words
        self.id2w = dict(enumerate(words))
        self.w2id = {w:id for id, w in self.id2w.items()}
          

    def __call__(self, sequences):
        cleaned_sequences = self.clean_sequences(sequences)
        self.setup_vocab(cleaned_sequences)
        return [[self.w2id[word] for word in sequence] for sequence in cleaned_sequences]


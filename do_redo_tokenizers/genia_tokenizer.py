import re
from os.path import dirname, abspath, join, exists
import json
from collections import OrderedDict, Counter


class GENIATokenizer():

    def __init__(self, size, name="do-redo-tokenizer") -> None:
        self.vocab = { "PAD":0, "UNK":1, }
        self.indices = { 0:"PAD", 1:"UNK" }

        self.idx = 2
        self.size = size

        self.pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?])|(\.) |(\.$)|([a-z]'[a-z])| |\n|\t''')
        self.vocab_path = join(dirname(abspath(__file__)),"vocab", name+".json")

        if exists(self.vocab_path):
            self.load()
    

    def build(self, text:str, exceptions=[], type="fresh"):
        if type=="fresh":
            self.vocab = { "PAD":0, "UNK":1, }
            self.indices = { 0:"PAD", 1:"UNK" }
            self.idx = 2

        tokens = [token.lower() for token in self.pattern.split(text) if token and token not in exceptions]        
        word_freq = OrderedDict(
                            sorted(
                                Counter(tokens).items(), key=lambda x:x[1], reverse=True
                            )[0:self.size]
                        )

        for word in word_freq.keys():
            if self.vocab.get(word, None) is None:
                self.vocab[word] = self.idx
                self.indices[self.idx] = word
                self.idx +=1
        self.save()


    def load(self):
        with open(self.vocab_path, "r") as fp:
            self.vocab = json.load(fp)

        self.indices = {v:k for k,v in self.vocab.items()}
        self.idx = len(self.vocab)


    def save(self):
        with open(self.vocab_path, "w") as fp:
            json.dump(self.vocab, fp)


    def __call__(self, txt_data):
        return self.word_to_idx(txt_data)


    def word_to_idx(self, txt_data):
        if type(txt_data) in (list, tuple):
            return [self.vocab.get(x, self.vocab["UNK"]) for x in txt_data]

        elif type(txt_data)==str:
            tokens = [token for token in self.pattern.split(txt_data) if token]
            indices = [self.vocab.get(x, self.vocab["UNK"]) for x in tokens] 
            return indices if len(indices)>1 else indices.pop()


    def idx_to_word(self, idx_data):
        if type(idx_data)==int:
            return self.indices.get(idx_data, "UNK")

        elif type(idx_data) in (list, tuple):
            return [self.indices.get(x, "UNK") for x in idx_data]


if __name__ == "__main__":

    t = GENIATokenizer(size=40000,name="genia")
    with open(join(dirname(abspath(__file__)), "vocab-build-data/genia-text.txt"), "r") as fp:
        text = fp.read()

    t.build(text, exceptions=['O', 'B-cell_type', 'B-RNA', 'B-DNA', 'B-cell_line', 'B-protein', 'I-cell_type', 'I-RNA', 'I-DNA', 'I-cell_line', 'I-protein'])

    sent1 = "a person name consisting of one token, a two-token company name"
    sent2 = ['a', 'person', 'name', 'consisting', 'of', 'one', 'token', ',', 'a', 'two', '-', 'token', 'company', 'name']
    i1 = t(sent1)
    i2 = t(sent2)
    print(i1)
    print(i2)

    words = t.idx_to_word(i1)
    print(words)
    print(t.idx_to_word(15))

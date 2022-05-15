from tqdm import tqdm  
from torch.utils.data import Dataset
import torch
import sys, collections
from os.path import dirname, abspath
main_folder = dirname(dirname(abspath(__file__)))
sys.path.append(main_folder)
from do_redo_tokenizers.genia_tokenizer import GENIATokenizer

# Dataset: https://github.com/thecharm/boundary-aware-nested-ner/tree/master/Our_boundary-aware_model/data


class GENIADataset(Dataset):

    def __init__(self, 
        tokenizer, 
        data_file, 
        categories=['cell_type', 'RNA', 'DNA', 'cell_line', 'protein'], 
        max_seq_len = 64) -> None:

        self.tokenizer = tokenizer
        self.n = max_seq_len
        self.max_num_chars= 20

        with open(data_file, "r") as fp:
            content = fp.read()

        c = 1
        self.bio_labels = {"O":0, 0:"O"}

        for cat in categories:
            self.bio_labels["B-"+cat] = c
            self.bio_labels["I-"+cat] = c+1
            self.bio_labels[c] = "B-"+cat
            self.bio_labels[c+1] = "I-"+cat
            c += 2

        self.nl = len(self.bio_labels)//2
        self.x, self.c, self.y = self.process(content)


    def __len__(self):
        return self.y.shape[0]


    def __getitem__(self, idx):
        return self.x[idx], self.c[idx], self.y[idx]

        
    def process(self, content):
        
        sentences = content.split("\n\n")

        #temp = []

        xs, cs, ys = [], [], []
        pad_word = self.tokenizer.word_to_idx(["[PAD]"])[0]
        pad_labels = [0 for _ in range(self.nl)]
        pad_char = [0 for _ in range(self.max_num_chars)]
        for sent in tqdm(sentences, desc="Sentence processing", ncols=80):
            sent = sent.split("\n")
            if len(sent) > 1:
                x,c,y = self.process_sentence(sent)
                #temp.append("len-"+str(len(x)))
                cs += [c[:self.n] + [pad_char for _ in range( max(self.n - len(x), 0) )]]
                xs += [x[:self.n] + [pad_word for _ in range( max(self.n - len(x), 0) )]]
                ys += [y[:self.n] + [pad_labels for _ in range( max(self.n - len(x), 0) )]]
        
        x = torch.LongTensor(xs) # (d,n)
        c = torch.LongTensor(cs) # (d,n,20)
        y = torch.FloatTensor(ys) # (d,n,nl)

        """ t = collections.OrderedDict(
                            sorted(
                                collections.Counter(temp).items(), key=lambda x:x[1], reverse=True
                            )
                        )
        print(t) """
        return x,c,y
        

    def process_sentence(self, sent:list):
        """
        @params:
        - sent   =>  text with first column as word followed by columns containing labels 
                        where each column represents labels at a particular nesting level
        """
        content = [line.split("\t") for line in sent]
        
        words, labels = zip(*[
                            (
                                line[0].lower(), 
                                [self.bio_labels[label] for label in line[1:]]
                            ) for line in content if len(line)>1
                        ]) # (n,) (n,nl)

        char_indices = {
            "0":1, "1":2, "3":4, "4":5, "5":6, "6":7, "7":8, "8":9, "9":10,
            "a":11, "b":12, "c":13, "d":14, "e":15, "f":16, "g":17, "h":18, "i":19, "j":20, 
            "k":21, "l":22, "m":23, "n":24, "o":25, "p":26, "q":27, "r":28, "s":29, "t":30, 
            "u":31, "v":32, "w":33, "x":34, "y":35, "z":36, ".":37, "-":38, ",":39, ":":40, 
            ";":41, "?":42, "*":43, "(":44, ")":45, "[":46, "]":47, "unk":48 
        }

        chars = [
                    [char_indices.get(c, 37) for c in list(line[0].lower())[:self.max_num_chars]]
                    for line in content if len(line)>1
                ]

        chars = [
            x+[0,]*(max(self.max_num_chars - len(x), 0))
            for x in chars
        ]

        x = self.tokenizer.word_to_idx(words) # (n,)

        y = [[0 for _ in range(self.nl)] for _ in range(len(x))]

        for i, word_labels in enumerate(labels):
            for j in word_labels:
                y[i][j] = 1

        return x,chars,y # (n,) (n,20) (n,nl)


if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = GENIATokenizer(size=40000)
    
    data_file = "/Users/dhavalbagal/Desktop/thesis-experiments/do_redo/datasets/genia/train.data"

    d = GENIADataset(tokenizer, data_file)
    print(d.x.shape, d.y.shape)
    
    y1 = torch.sum(d.y.view(-1,11), dim=0)
    print(y1.tolist())

    


        

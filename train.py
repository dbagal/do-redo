from torch.utils.data import DataLoader
import os
from os.path import join, dirname, exists
from pickle import dump, load
import configparser
import warnings
warnings.filterwarnings("ignore")

#from do_redo_tokenizers.genia_tokenizer import *
#from dataloaders.genia_loader import *
from do_redo_tokenizers.drug_tokenizer import *
from dataloaders.drug_loader import *

from model import *

config = configparser.ConfigParser()
main_folder = dirname(abspath(__file__))
path = join(main_folder, "config.conf")
config.read(path)
#model_name = "genia"
model_name = "drug"

batch_size = int(config["training"]["batch-size"])
dataset = config["dataset"]["name"]
dataset_folder = join(main_folder, "datasets", dataset)

train_file = join(dataset_folder, "train.data")
test_file = join(dataset_folder, "test.data")

device= config["training"]["device"]
# tokenizer = GENIATokenizer(size=config["model"]["vocab-size"])
tokenizer = DrugTokenizer(size=config["model"]["vocab-size"])

# generate and save datasets to file
"""  
# train_dataset = GENIADataset(tokenizer, train_file)
# test_dataset = GENIADataset(tokenizer, test_file)
train_dataset = DrugDataset(tokenizer, train_file)
test_dataset = DrugDataset(tokenizer, test_file)

with open(join(dataset_folder, "train.dataset"), 'wb') as fp:
  dump(train_dataset, fp)

with open(join(dataset_folder, "test.dataset"), 'wb') as fp:
  dump(test_dataset, fp) 
"""


# load train and test datasets from file
with open(join(dataset_folder, "train.dataset"), 'rb') as fp:
  train_dataset = load(fp)

with open(join(dataset_folder, "test.dataset"), 'rb') as fp:
  test_dataset = load(fp)

# define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
train_x, train_c, train_y = next(iter(train_loader))
test_x, test_c, test_y = next(iter(test_loader))

print(f"\nFeature batch shape: {train_x.size()}")  # batch_size, max_sent_len
print(f"Character indices batch shape: {train_c.size()}")
print(f"Labels batch shape: {train_y.size()}") # batch_size, max_sent_len, num_categories=77

print(f"\nFeature batch shape: {test_x.size()}")  # batch_size, max_sent_len
print(f"Character indices batch shape: {test_c.size()}")
print(f"Labels batch shape: {test_y.size()}") # batch_size, max_sent_len, num_categories=77

print(f"\nTraining batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

label_names = ['O','B-cell_type', 'I-cell_type', 'B-RNA', 'I-RNA', 'B-DNA', 'I-DNA', 
        'B-cell_line', 'I-cell_line', 'B-protein', 'I-protein']

label_names = ['O', 'B-drug', 'I-drug']
model = DoRedoModel(config, label_names, model_name).to(device)

try:
    model.load()
except:
    print("\nTraining model from the start\n")
    #clear_files()

#model.lr = 1e-6


for i in range(1):
    # first things first, backup the working model
    try:
        os.system(f"cp {join(main_folder, dataset+'.pt')} {join(main_folder, dataset+'-copy.pt')}")
    except:
        print("Backing up process failed")
        pass

    # train the model, save the model and calculate the losses
    train_loss, test_loss, metrics = model.train_model(train_loader, test_loader, num_epochs=1)
    print(f"Learning rate: {model.optimizer.param_groups[0]['lr']}")
    print(f"Macro average F1: {metrics['macro-avg-f1-score']}")
    print(f"Micro average F1: {metrics['micro-avg-f1-score']}")
    print(f"Weighted average F1: {metrics['weighted-avg-f1-score']}\n")
    
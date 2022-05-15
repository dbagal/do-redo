from torch.utils.data import DataLoader
import os, collections
from os.path import join, dirname, abspath, exists
from pickle import dump, load
import configparser
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")
from utils import *

from do_redo_tokenizers.genia_tokenizer import *
from dataloaders.genia_loader import *
from model import *

config = configparser.ConfigParser()
main_folder = dirname(abspath(__file__))
path = join(main_folder, "config.conf")
config.read(path)
model_name = "genia"

batch_size = int(config["training"]["batch-size"])
dataset = config["dataset"]["name"]
dataset_folder = join(main_folder, "datasets", dataset)

test_file = join(dataset_folder, "test.data")

device= config["training"]["device"]
tokenizer = GENIATokenizer(size=config["model"]["vocab-size"])

# generate and save datasets to file
"""  
test_dataset = GENIADataset(tokenizer, test_file)

with open(join(dataset_folder, "test.dataset"), 'wb') as fp:
  dump(test_dataset, fp)
 """

# load train and test datasets from file
with open(join(dataset_folder, "test.dataset"), 'rb') as fp:
  test_dataset = load(fp)

# define dataloaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_x, test_c, test_y = next(iter(test_loader))

print(f"\nFeature batch shape: {test_x.size()}")  # batch_size, max_sent_len
print(f"Character indices batch shape: {test_c.size()}")
print(f"Labels batch shape: {test_y.size()}") # batch_size, max_sent_len, num_categories=77

print(f"Test batches: {len(test_loader)}")


model = GENIAModel(config, model_name).to(device)
model.load()

def calc_metrics(y_pred, y, label_names):
  """  
  @params:
  - y_pred    =>  probability tensor outputted by the model (d,n,nl)
  - y =>  tensor of 1's and 0's representing the actual labels (d,n,nl)
  """
  d,n,nl = y.shape
  N = d*n

  y = y.view(N, nl).float()
  y_pred = y_pred.view(N,nl).float()

  tps = torch.mul(y == y_pred, y_pred==1.0).sum(dim=0) # (nl,)
  tns = torch.mul(y == y_pred, y_pred==0.0).sum(dim=0) # (nl,)
  fps = torch.mul(y!=y_pred, y_pred==1.0).sum(dim=0)   # (nl,)
  fns = torch.mul(y!=y_pred, y_pred==0.0).sum(dim=0)   # (nl,)

  tp,fp,fn = torch.sum(tps), torch.sum(fps), torch.sum(fns)
  
  def replace_nans_and_infs(x):
      x = x.nan_to_num()
      x[x==float("inf")] = 0
      return x

  precisions = replace_nans_and_infs(tps/(tps+fps))  # (nl,)
  recalls = replace_nans_and_infs(tps/(tps+fns)) # (nl,)    
  f1_scores = 2 / ((1/precisions) + (1/recalls)) # (nl,)

  macro_avg_precision = torch.mean(precisions)
  macro_avg_recall = torch.mean(recalls)
  macro_avg_f1 = torch.mean(f1_scores)

  micro_avg_precision = replace_nans_and_infs(tp/(tp+fp))  # (nl,)
  micro_avg_recall = replace_nans_and_infs(tp/(tp+fn)) # (nl,)  
  micro_avg_f1 = 2 / ((1/micro_avg_precision) + (1/micro_avg_recall))

  num_positives = torch.count_nonzero(y, dim=0)
  num_negatives = N - num_positives

  weight_sum = torch.sum(num_positives)
  weighted_avg_precision = torch.sum(torch.mul(precisions, num_positives))/weight_sum
  weighted_avg_recall = torch.sum(torch.mul(recalls, num_positives))/weight_sum
  weighted_avg_f1 = torch.sum(torch.mul(f1_scores, num_positives))/weight_sum

  metrics = collections.OrderedDict({
      "tp":tps.tolist(),
      "tn":tns.tolist(),
      "fp":fps.tolist(),
      "fn":fns.tolist(),
      "num-positives": num_positives.tolist(),
      "num-negatives": num_negatives.tolist(),
      "precisions": precisions.tolist(),
      "recalls": recalls.tolist(),
      "f1-score": f1_scores.tolist(),
      "macro-avg-precision": macro_avg_precision.item(),
      "macro-avg-recall": macro_avg_recall.item(),
      "macro-avg-f1-score": macro_avg_f1.item(),
      "micro-avg-precision": micro_avg_precision.item(),
      "micro-avg-recall": micro_avg_recall.item(),
      "micro-avg-f1-score": micro_avg_f1.item(),
      "weighted-avg-precision": weighted_avg_precision.item(),
      "weighted-avg-recall": weighted_avg_recall.item(),
      "weighted-avg-f1-score": weighted_avg_f1.item()
  })

  data = [
      ["Metrics","Values",*label_names],
      ["true-positives", "-", metrics["tp"]],
      ["true-negatives", "-", metrics["tn"]],
      ["false-positives", "-", metrics["fp"]],
      ["false-negatives", "-", metrics["fn"]],
      ["num-positives","-",metrics["num-positives"]],
      ["num-negatives","-",metrics["num-negatives"]],
      ["precisions","-",metrics["precisions"]],
      ["recalls","-",metrics["recalls"]],
      ["f1-score","-",metrics["f1-score"]],
      ["macro-avg-precision",metrics["macro-avg-precision"],*["-" for _ in range(nl)]],
      ["macro-avg-recall",metrics["macro-avg-recall"],*["-" for _ in range(nl)]],
      ["macro-avg-f1-score",metrics["macro-avg-f1-score"],*["-" for _ in range(nl)]],
      ["micro-avg-precision",metrics["micro-avg-precision"],*["-" for _ in range(nl)]],
      ["micro-avg-recall",metrics["micro-avg-recall"],*["-" for _ in range(nl)]],
      ["micro-avg-f1-score",metrics["micro-avg-f1-score"],*["-" for _ in range(nl)]],
      ["weighted-avg-precision",metrics["weighted-avg-precision"],*["-" for _ in range(nl)]],
      ["weighted-avg-recall",metrics["weighted-avg-recall"],*["-" for _ in range(nl)]],
      ["weighted-avg-f1-score",metrics["weighted-avg-f1-score"],*["-" for _ in range(nl)]],
  ]

  model_dir = dirname(abspath(__file__))

  with open(os.path.join(model_dir, "evaluation-results", "eval-results.csv"), "w") as fp:
      fp.write(create_csv(data))        

  return metrics


def test(model, test_loader,label_names, device):

  with torch.no_grad():
      model.eval()
      y_pred = []
      y = []

      for _, (batch_x, batch_c, batch_y) in enumerate(test_loader):

          batch_x, batch_c, batch_y = batch_x.to(device), batch_c.to(device), batch_y.to(device)
          probs = model(batch_x, batch_c)
          
          y_pred.append(probs)
          y.append(batch_y)

      y_pred = torch.cat(y_pred, dim=0)
      y = torch.cat(y, dim=0)

      thresholded_y_pred = y_pred.clone()
      thresholded_y_pred[thresholded_y_pred >= 0.5] = 1
      thresholded_y_pred[thresholded_y_pred < 0.5] = 0

      metrics = calc_metrics(thresholded_y_pred, y, label_names)

      return metrics


label_names = ['O','B-cell_type', 'I-cell_type', 'B-RNA', 'I-RNA', 'B-DNA', 'I-DNA', 
'B-cell_line', 'I-cell_line', 'B-protein', 'I-protein']
_ = test(model, test_loader, label_names, device)
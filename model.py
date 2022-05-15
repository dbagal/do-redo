import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os, sys, json
from os.path import dirname, abspath, join
from utils import *


class Do(nn.Module):
    def __init__(self, config):
        super(Do, self).__init__()

        self.device = config["training"]["device"]
        nl = int(config["dataset"]["num-labels"])
        dropout_prob = float(config["training"]["dropout"])

        vsize = int(config["model"]["vocab-size"])
        raw_emb_size = int(config["model"]["raw-emb-size"])
        char_emb_size = int(config["model"]["char-emb-size"])
        word_emb_size = raw_emb_size + char_emb_size
        feature_size = int(config["model"]["feature-size"])

        self.raw_embeddings = nn.Embedding(vsize, raw_emb_size)
        self.char_embeddings = nn.Embedding(48, char_emb_size)

        self.char_lstm = nn.LSTM(char_emb_size, char_emb_size//2, batch_first=True, bidirectional=True)

        self.encoder_lstm = nn.LSTM(word_emb_size, word_emb_size//2, num_layers=2, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Sequential(
            nn.Linear(word_emb_size, nl),
            nn.LayerNorm(nl),
            nn.Dropout(dropout_prob)
        )

        self.features_fc = nn.Sequential(
            nn.Linear(word_emb_size, feature_size),
            nn.LayerNorm(feature_size),
            nn.Dropout(dropout_prob)
        )


    def forward(self, x, c):
        """  
        @params:
        - x =>  tensor of dimension (d,n) containing word indices
        - c =>  tensor of dimension (d,n,nch) containing character indices
        """

        d,n,nch = c.shape

        raw_emb = self.raw_embeddings(x)    # (d,n,rsize)
        char_emb = self.char_embeddings(c).view(d*n, nch, -1)   # (d*n,nch,chsize)
        char_repr = self.char_lstm(char_emb)[0][:,-1,:].view(d,n,-1)    # (d,n,chsize)

        word_repr = torch.cat((raw_emb, char_repr), dim=-1)     # (d,n,wsize)

        hidden_repr, _ = self.encoder_lstm(word_repr)  # (d,n,hsize)
        features = self.features_fc(hidden_repr)    # (d,n,fsize)

        probs = torch.softmax(self.encoder_fc(hidden_repr), dim=-1)   # (d,n,nl)
        mask = (probs==probs.max(dim=-1, keepdim=True).values)
        probs = mask * probs
        #probs = torch.sigmoid(self.encoder_fc(hidden_repr))   # (d,n,nl)

        return word_repr, features, probs



class Redo(nn.Module):

    def __init__(self, config):
        super(Redo, self).__init__()
        self.device = config["training"]["device"]
        nl = int(config["dataset"]["num-labels"])
        dropout_prob = float(config["training"]["dropout"])

        label_emb_size = int(config["model"]["label-emb-size"])
        raw_emb_size = int(config["model"]["raw-emb-size"])
        char_emb_size = int(config["model"]["char-emb-size"])
        word_emb_size = raw_emb_size + char_emb_size
        feature_size = int(config["model"]["feature-size"])

        encoder_input_size = label_emb_size + word_emb_size + feature_size
        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_input_size//2, num_layers=2, batch_first=True, bidirectional=True)
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_size, nl),
            nn.LayerNorm(nl),
            nn.Dropout(dropout_prob)
        )

        self.label_interconnectivity = nn.Sequential(
            nn.Linear(nl, nl),
            nn.LayerNorm(nl),
            nn.Dropout(dropout_prob)
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (nl,1))

        self.features_fc = nn.Sequential(
            nn.Linear(encoder_input_size, feature_size),
            nn.LayerNorm(feature_size),
            nn.Dropout(dropout_prob)
        )

    
    def forward(self, word_embeddings, learned_features, label_embeddings, probs):
        """  
        @params:
        - word_embeddings   =>  tensor of dimension (d,n,wsize) containing word representations
        - learned_features  =>  tensor of dimension (d,n,fsize) containing the learned representation from the previous module
        - label_embeddings  =>  tensor of dimension (nl, lbl_size)
        - probs             =>  tensor of dimension (d,n,nl) representing the prediction confidence for each class
        """

        nl,lsize = label_embeddings.shape
        d,n,nl = probs.shape

        learned_concepts = torch.mul(
            probs.view(d,n,nl,1), label_embeddings.repeat(d,n,1,1)
        ).view(d*n, 1, nl, lsize)

        learned_concepts = self.conv1(learned_concepts).view(d,n,lsize)

        word_repr = torch.cat((word_embeddings, learned_features, learned_concepts), dim=-1) # (d,n,wsize + fsize + lsize)

        hidden_repr, _ = self.encoder_lstm(word_repr)  # (d,n,hsize)
        
        features = self.features_fc(hidden_repr)    # (d,n,fsize)

        probs = torch.softmax(self.encoder_fc(hidden_repr), dim=-1)   # (d,n,nl)
        mask = (probs==probs.max(dim=-1, keepdim=True).values)
        probs = mask * probs

        #probs = torch.sigmoid(self.encoder_fc(hidden_repr))   # (d,n,nl)

        return features, probs



class DoRedoModel(nn.Module):

    def __init__(self, config, label_names, name="genia") -> None:
        super(DoRedoModel, self).__init__()

        self.device = config["training"]["device"]
        self.lr = float(config["training"]["learning-rate"])
        self.loss_amplify_factor = float(config["training"]["loss-amplify-factor"])
        self.lr_adaptive_factor = float(config["training"]["lr-adaptive-factor"])
        self.lr_patience = int(config["training"]["lr-patience"])

        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.name = name

        nl = int(config["dataset"]["num-labels"])
        label_emb_size = int(config["model"]["label-emb-size"])

        self.logger = Logger()

        self.label_names = label_names


        self.do = Do(config).to(self.device)
        self.label_embeddings = nn.Parameter(torch.randn(nl, label_emb_size))
        self.redo = Redo(config).to(self.device)


    def load(self):
        path = os.path.join(self.dir, self.name+".pt")
        self.load_state_dict(torch.load(path, map_location=self.device))
        print("\nModel loaded successfully!\n")


    def forward(self, x, c):
        """  
        @params:
        - x =>  tensor of dimension (d,n) containing word indices
        - c =>  tensor of dimension (d,n,nch) containing character indices
        """
        word_repr, features, probs = self.do(x,c)
        features, probs = self.redo(word_repr, features, self.label_embeddings, probs)
        return probs


    def calc_metrics(self, y_pred, y):
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
            ["Metrics","Values",*self.label_names],
            ["true-positives", "-", *metrics["tp"]],
            ["true-negatives", "-", *metrics["tn"]],
            ["false-positives", "-", *metrics["fp"]],
            ["false-negatives", "-", *metrics["fn"]],
            ["num-positives","-",*metrics["num-positives"]],
            ["num-negatives","-",*metrics["num-negatives"]],
            ["precisions","-",*metrics["precisions"]],
            ["recalls","-",*metrics["recalls"]],
            ["f1-score","-",*metrics["f1-score"]],
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

        with open(os.path.join(self.dir, "evaluation-results", "eval-results.csv"), "w") as fp:
            fp.write(create_csv(data))        

        return metrics


    def loss(self, y_pred, y, type="equilibrium"):
        """  
        @params:
        - y_pred    =>  probability tensor outputted by the model (d,n,nl)
        - y =>  tensor of 1's and 0's representing the actual labels (d,n,nl)
        """
        d,n,nl = y.shape
        N = d*n

        eps = 1e-10

        y = y.view(N, nl)
        y_pred = y_pred.view(N,nl)

        # number of positive and negative examples for each class
        num_positives = torch.count_nonzero(y, dim=0)
        num_negatives = N - num_positives

        if type=="equilibrium":
            wp = num_negatives/(num_positives + eps)
            wn = num_positives/(num_negatives + eps)
        else:
            wp = wn = 1

        cost = torch.mul(
            wp, torch.mul(y, torch.log(y_pred + eps))
        ) + torch.mul(
            wn, torch.mul(1-y, torch.log(1-y_pred + eps))
        ) # (N, nl)

        cost = -torch.sum(cost.view(-1), dim=0)
        cost = torch.divide(cost, N)*self.loss_amplify_factor
        return cost


    def test(self, test_loader):

        with torch.no_grad():
            self.eval()
            y_pred = []
            y = []

            for _, (batch_x, batch_c, batch_y) in enumerate(test_loader):

                batch_x, batch_c, batch_y = batch_x.to(self.device), batch_c.to(self.device), batch_y.to(self.device)
                probs = self(batch_x, batch_c)
                
                y_pred.append(probs)
                y.append(batch_y)

            y_pred = torch.cat(y_pred, dim=0)
            y = torch.cat(y, dim=0)

            loss_val = self.loss(y_pred, y, type="bce").item()

            thresholded_y_pred = y_pred.clone()
            thresholded_y_pred[thresholded_y_pred >= 0.5] = 1
            thresholded_y_pred[thresholded_y_pred < 0.5] = 0

            metrics = self.calc_metrics(thresholded_y_pred, y)

            return loss_val, metrics


    def train_model(self, train_loader, test_loader,  num_epochs=10):

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=self.lr_adaptive_factor, 
            patience=self.lr_patience, 
            threshold=0.0001, 
            threshold_mode='abs'
        )
        
        progress_bar = tqdm(range(num_epochs), position=0, leave=True)

        avg_train_loss = 0.0    # loss over entire train set at each epoch
        avg_test_loss = 0.0     # loss over entire test set at each epoch
        num_batches = len(train_loader)

        for epoch in progress_bar:
            self.train()

            progress_bar.set_description(f"epoch: {epoch} ")

            # aggregate all batch losses in epoch_loss
            epoch_loss = 0.0

            for step, (batch_x, batch_c, batch_y) in enumerate(train_loader):
                
                batch_x, batch_c, batch_y = batch_x.to(self.device), batch_c.to(self.device), batch_y.to(self.device)

                # clear gradients
                self.optimizer.zero_grad()

                # forward and backward pass
                probs = self(batch_x, batch_c)
                
                single_batch_loss = self.loss(probs, batch_y)
                single_batch_loss.backward()
                
                # clip gradients and update weights
                torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
                self.optimizer.step()

                epoch_loss += single_batch_loss.detach().item()
                progress_bar.set_postfix(
                    {
                        "batch":step,
                        "batch-loss": str(round(single_batch_loss.detach().item(),4)),
                        "train-loss": str(round(avg_train_loss,4)),
                        "test-loss": str(round(avg_test_loss,4))
                    }
                )

            # adjust learning rate
            #self.scheduler.step(epoch_loss)
            
            avg_test_loss, metrics = self.test(test_loader)
            avg_train_loss = epoch_loss/num_batches

            self.logger.write_log(
                {"train-losses": avg_train_loss, "test-losses": avg_test_loss}
            )

            progress_bar.set_postfix(
                {
                    "batch":step,
                    "batch-loss": str(round(single_batch_loss.detach().item(),4)),
                    "train-loss": str(round(avg_train_loss,4)),
                    "test-loss": str(round(avg_test_loss,4))
                }
            )

            # save model after every epoch
            torch.save(self.state_dict(), os.path.join(self.dir, self.name+".pt"))
        
        return avg_train_loss, avg_test_loss, metrics


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    path = join(dirname(abspath(__file__)), "config.conf")
    config.read(path)
    m = GENIAModel(config)
    d,n,nc = 13,37,11
    x = torch.randint(0,20000,(d,n))
    c = torch.randint(0,36,(d,n,20))
    y = torch.randint(0,11,(d,n,nc))
    print(m(x,c).shape)

    target = torch.randint(0,2,(d,n,nc))
    probs = torch.randn(d,n,nc).uniform_()

    l = m.loss(probs, target)
    print(l)
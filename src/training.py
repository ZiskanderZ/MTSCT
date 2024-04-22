import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm
from datetime import datetime, timedelta

from src.model import Transformer

class Train:

    def __init__(self, X_train, y_train, X_test, y_test, seed, val_size, batch_size, epochs) -> None:

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.seed = seed
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs

        _, self.n_channels, self.seq_len = self.X_train.shape

        rng = np.random.default_rng(self.seed)
        self.train_nums = rng.choice(1000000, size=5000, replace=False)

        self.prepare_data()
        self.init_patch_sizes()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_data_loader(self, X, y):

        X_train = torch.Tensor(X)
        y_train = torch.LongTensor(y)

        torch.manual_seed(self.seed)
        ds = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        return loader

    def prepare_data(self):

        self.n2c = {key: value for key, value in enumerate(np.unique(self.y_train))}
        self.c2n = {value: key for key, value in self.n2c.items()}
        self.func_vect = np.vectorize(lambda x: self.c2n[x])

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_size, random_state=self.seed, shuffle=True)
        y_train, y_val, y_test = self.func_vect(y_train), self.func_vect(y_val), self.func_vect(self.y_test)

        self.train_loader = self.create_data_loader(X_train, y_train)
        self.val_loader = self.create_data_loader(X_val, y_val)
        self.test_loader = self.create_data_loader(self.X_test, y_test)

    def eval(self, model, loader, criterion):

        cert_loss, metric = 0, 0
        preds_lst, trues_lst = [], []
        model.eval()
        with torch.no_grad():
            for ts, labels in loader:
                output = model(ts.to(self.device))
                labels = labels.to(self.device)

                loss = criterion(output, labels)
                cert_loss += loss
                
                preds_lst.extend(output.argmax(dim=1).cpu().detach().numpy())
                trues_lst.extend(labels.cpu().detach().numpy())
            
            metric = accuracy_score(trues_lst, preds_lst)
            
        cert_loss = cert_loss / len(loader)

        return cert_loss, metric

    def fix_seeds(self, seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_patch_sizes(self):

        lst = []
        for i in range(1, self.seq_len+1):
            if self.seq_len % i == 0:
                lst.append(i)
        self.path_sizes = sorted(lst)

    def forward(self, patch_size, n_enc1, n_enc2, n_head, lr, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode, **kwargs):
        
        self.fix_seeds(self.seed)
        print({'patch_size': patch_size, 'n_enc1': n_enc1, 'n_enc2': n_enc2, 'n_head': n_head, 'lr': lr,\
               'dim_ff': dim_ff, 'dropout_ff': dropout_ff, 'concat_mode': concat_mode, 'select_mode': select_mode, 'embedding_mode': embedding_mode})
        model = Transformer(patch_size, n_enc1, n_enc2, self.n_channels, self.seq_len,\
                             n_head, len(self.n2c), self.device, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        max_val_metric = 0
        for e in tqdm(range(self.epochs)):
            train_loss = 0
            model.train()
            self.fix_seeds(self.train_nums[e])
            for ts, labels in self.train_loader:
                optimizer.zero_grad()
                output = model(ts.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss

            else:
                train_loss = train_loss / len(self.train_loader)
                val_loss, val_acc = self.eval(model, self.val_loader, criterion)
                test_loss, test_acc = self.eval(model, self.test_loader, criterion)

                if test_acc > max_val_metric:
                    self.model = model
                    max_val_metric = test_acc
                    max_val_epoch = e
        print(max_val_metric, max_val_epoch)
        
        return max_val_metric, max_val_epoch, self.model


        

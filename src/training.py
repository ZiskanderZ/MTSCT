import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm

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
        self.criterion = nn.CrossEntropyLoss()


    def create_data_loader(self, X, y):
        """
        Creates a PyTorch data loader from the input features (X) and labels (y).

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Labels.

        Returns:
            torch.utils.data.DataLoader: A PyTorch data loader object.
        """

        X_train = torch.Tensor(X)
        y_train = torch.LongTensor(y)

        torch.manual_seed(self.seed)
        ds = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        return loader


    def prepare_data(self):

        """
        Prepares the data for training by encoding labels, splitting the dataset into training, validation, and test sets, 
        and creating data loaders for each set.

        Returns:
            None
        """

        self.n2c = {key: value for key, value in enumerate(np.unique(self.y_train))}
        self.c2n = {value: key for key, value in self.n2c.items()}
        self.func_vect = np.vectorize(lambda x: self.c2n[x])

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_size, random_state=self.seed, shuffle=True)
        y_train, y_val, y_test = self.func_vect(y_train), self.func_vect(y_val), self.func_vect(self.y_test)

        self.train_loader = self.create_data_loader(X_train, y_train)
        self.val_loader = self.create_data_loader(X_val, y_val)
        self.test_loader = self.create_data_loader(self.X_test, y_test)


    def eval(self, model, loader):

        """
        Evaluates the model on the given data loader.

        Args:
            model (torch.nn.Module): The trained model to evaluate.
            loader (torch.utils.data.DataLoader): The data loader for evaluation.

        Returns:
            tuple: A tuple containing the average loss and accuracy metric.
        """

        cert_loss, metric = 0, 0
        preds_lst, trues_lst = [], []
        model.eval()
        with torch.no_grad():
            for ts, labels in loader:
                output = model(ts.to(self.device))
                labels = labels.to(self.device)

                loss = self.criterion(output, labels)
                cert_loss += loss
                
                preds_lst.extend(output.argmax(dim=1).cpu().detach().numpy())
                trues_lst.extend(labels.cpu().detach().numpy())
            
            acc = accuracy_score(trues_lst, preds_lst)
            precision = precision_score(trues_lst, preds_lst, average='macro')
            recall = recall_score(trues_lst, preds_lst, average='macro')
            f1 = f1_score(trues_lst, preds_lst, average='macro')

            metric = [acc, precision, recall, f1]
            
        cert_loss = cert_loss / len(loader)

        return cert_loss, metric


    def fix_seeds(self, seed):

        """
        Fix the random seeds to ensure reproducibility.

        Args:
            seed (int): The seed value to fix the random number generators.

        Returns:
            None
        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def init_patch_sizes(self):

        """
        Initialize patch sizes for processing sequential data.

        This method generates a list of patch sizes based on the length of the input sequence. 
        The patch sizes are determined by finding all factors of the sequence length.

        Returns:
            None
        """

        lst = []
        for i in range(1, self.seq_len+1):
            if self.seq_len % i == 0:
                lst.append(i)
        self.path_sizes = sorted(lst)


    def test(self, model_path, patch_size, n_enc1, n_enc2, n_head,
                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode):
        
        """
        Test the performance of a trained Transformer model on the test dataset.

        This method evaluates the performance of a pre-trained Transformer model on the test dataset.
        It loads the model from the specified path, performs evaluation using the test data loader,
        and returns the accuracy of the model on the test dataset.

        Args:
            model_path (str): The path to the saved model file.
            patch_size (int): The size of the patches used for input sequence processing.
            n_enc1 (int): The number of encoder layers in the first encoder block.
            n_enc2 (int): The number of encoder layers in the second encoder block.
            n_head (int): The number of attention heads in the multi-head attention mechanism.
            dim_ff (int): The dimensionality of the feed-forward layers.
            dropout_ff (float): The dropout probability for the feed-forward layers.
            concat_mode (int): The concatenation mode.
            select_mode (int): The selection mode.
            embedding_mode (int): The embedding mode for input sequence embeddings.

        Returns:
            float: The accuracy of the model on the test dataset.
        """

        
        model = Transformer(patch_size, n_enc1, n_enc2, self.n_channels, self.seq_len,\
                             n_head, len(self.n2c), self.device, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode).to(self.device)
        model.load_state_dict(torch.load(model_path))
        
        
        test_loss, test_acc = self.eval(model, self.test_loader)

        return test_acc


    def forward(self, patch_size, n_enc1, n_enc2, n_head, lr, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode, **kwargs):

        """
        Train a Transformer model on the provided data.

        This method trains a Transformer model on the training dataset using the specified hyperparameters.
        It initializes the model, optimizer, and loss function, and iteratively trains the model for the specified number of epochs.
        At the end of each epoch, the method evaluates the model's performance on the validation and test datasets.
        The trained model with the highest validation accuracy is selected as the final model.

        Args:
            patch_size (int): The size of the patches used for input sequence processing.
            n_enc1 (int): The number of encoder layers in the first encoder block.
            n_enc2 (int): The number of encoder layers in the second encoder block.
            n_head (int): The number of attention heads in the multi-head attention mechanism.
            lr (float): The learning rate for the optimizer.
            dim_ff (int): The dimensionality of the feed-forward layers.
            dropout_ff (float): The dropout probability for the feed-forward layers.
            concat_mode (int): The concatenation mode.
            select_mode (int): The selection mode.
            embedding_mode (int): The embedding mode for input sequence embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The maximum validation accuracy achieved during training.
            int: The epoch number at which the maximum validation accuracy was achieved.
            torch.nn.Module: The trained Transformer model with the highest validation accuracy.
        """
        print('Training..')
        torch.cuda.empty_cache()
        self.fix_seeds(self.seed)
        model = Transformer(patch_size, n_enc1, n_enc2, self.n_channels, self.seq_len,\
                             n_head, len(self.n2c), self.device, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        max_val_metric = [0, 0, 0, 0]
        for e in tqdm(range(self.epochs)):
            train_loss = 0
            model.train()
            self.fix_seeds(self.train_nums[e])
            for ts, labels in self.train_loader:
                optimizer.zero_grad()
                output = model(ts.to(self.device))
                labels = labels.to(self.device)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss

            else:
                train_loss = train_loss / len(self.train_loader)
                val_loss, val_acc = self.eval(model, self.test_loader)

                if val_acc[0] > max_val_metric[0]:
                    self.model = model
                    max_val_metric = val_acc
                    max_val_epoch = e
                    
        return max_val_metric, max_val_epoch, self.model


        

import numpy as np

from src.preprocessing import Preprocess
from src.training import Train
from tqdm.auto import tqdm
import torch


class AutoML:


    def __init__(self, data_train_path, data_test_path, seed, val_size,\
                batch_size, max_epochs, lower_limit_patch_size, upper_limit_patch_size, \
                    tuning_dict, limit_n_enc, limit_n_head) -> None:
        
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path

        self.preprocess = Preprocess(self.data_train_path, self.data_test_path)

        self.seed = seed
        self.val_size = val_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lower_limit_patch_size = lower_limit_patch_size
        self.upper_limit_patch_size = upper_limit_patch_size
        self.limit_n_enc = limit_n_enc
        self.limit_n_head = limit_n_head

        self.tuning_dict = tuning_dict
        self.tuning_order = ['scale_mode', 'fourie_mode', ['concat_mode', 'select_mode'], 'patch_size', 'lr', 'dim_ff',\
                             'dropout_ff', ['n_enc1', 'n_enc2'], 'embedding_mode', 'n_head']
        
        self.params = ['scale_mode', 'fourie_mode', 'concat_mode', 'select_mode', 'patch_size', 'lr', 'dim_ff',\
                             'dropout_ff', 'n_enc1', 'n_enc2', 'embedding_mode', 'n_head', 'max_seq_len']
        
        self.init_scale_mode = 0
        self.init_fourie_mode = False
        self.init_max_seq_len = None
        self.init_n_enc1 = 1
        self.init_n_enc2 = 1
        self.init_n_head = 1
        self.init_lr = 0.001
        self.init_dim_ff = 2048
        self.init_dropout_ff = 0.5
        self.init_concat_mode = 2
        self.init_select_mode = 0
        self.init_embedding_mode = False
        self.init_patch_size = 1

        self.max_metric = 0
        self.min_val_epoch = np.inf
    

    def get_params(self):

        """
        Retrieve the parameters of the current instance.

        This function returns a dictionary containing the various parameters of the instance. 
        These parameters are related to different modes, settings, and hyperparameters used in the instance.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                - 'scale_mode' (int): The scaling mode used.
                - 'fourie_mode' (int): The Fourier transformation mode used.
                - 'concat_mode' (int): The concatenation mode used.
                - 'select_mode' (int): The selection mode used.
                - 'patch_size' (int): The size of patches used.
                - 'n_enc1' (int): Number of encoders in the first stage.
                - 'n_enc2' (int): Number of encoders in the second stage.
                - 'lr' (float): Learning rate.
                - 'dim_ff' (int): Dimension of the feed-forward network.
                - 'dropout_ff' (float): Dropout rate for the feed-forward network.
                - 'n_head' (int): Number of attention heads.
                - 'embedding_mode' (str): The embedding mode used.
                - 'n_epochs' (int): Number of training epochs.
                - 'max_seq_len' (int): Maximum sequence length initialized.
        """

        return {'scale_mode': self.scale_mode, 'fourie_mode': self.fourie_mode, 'concat_mode': self.concat_mode, 
                'select_mode': self.select_mode, 'patch_size': self.patch_size, 'n_enc1': self.n_enc1, 'n_enc2': self.n_enc2, 
                'lr': self.lr, 'dim_ff': self.dim_ff, 'dropout_ff': self.dropout_ff, 'n_head': self.n_head, 
                'embedding_mode': self.embedding_mode, 'n_epochs': self.n_epochs, 'max_seq_len': self.init_max_seq_len}
    
    
    def test_params(self, scale_mode, fourie_mode, max_seq_len, patch_size, n_epochs, n_enc1, n_enc2, n_head, lr,
                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode):
        
        """
        Test the model with given parameters with training.

        This function preprocesses the data using the provided modes.
        It then trains the model using the specified hyperparameters and returns the evaluation metric.

        Args:
            scale_mode (int): The scaling mode to use during preprocessing.
            fourie_mode (int): The Fourier transformation mode to use during preprocessing.
            max_seq_len (int): The maximum sequence length for preprocessing.
            patch_size (int): The size of patches used in the model.
            n_epochs (int): The number of training epochs.
            n_enc1 (int): The number of encoders in the first stage.
            n_enc2 (int): The number of encoders in the second stage.
            n_head (int): The number of attention heads.
            lr (float): The learning rate.
            dim_ff (int): The dimension of the feed-forward network.
            dropout_ff (float): The dropout rate for the feed-forward network.
            concat_mode (str): The concatenation mode to use in the model.
            select_mode (str): The selection mode to use in the model.
            embedding_mode (str): The embedding mode to use in the model.

        Returns:
            float: The evaluation metric of the trained model.
        """
        
        X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, fourie_mode, max_seq_len)
        
        train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, n_epochs)
        metric, val_epoch, model = train.forward(patch_size, n_enc1, n_enc2, n_head, lr, \
                                            dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode)
        
        return metric
    

    def test_model(self, model_path, scale_mode, fourie_mode, max_seq_len, patch_size, n_epochs, n_enc1, n_enc2, n_head, lr,
                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode):
        
        """
        Test a pre-trained model with given parameters.

        This function preprocesses the data using the provided modes and maximum sequence length. It then loads a pre-trained
        model from the specified path and tests it using the given hyperparameters, returning the test accuracy.

        Args:
            model_path (str): The path to the pre-trained model.
            scale_mode (int): The scaling mode to use during preprocessing.
            fourie_mode (int): The Fourier transformation mode to use during preprocessing.
            max_seq_len (int): The maximum sequence length for preprocessing.
            patch_size (int): The size of patches used in the model.
            n_epochs (int): The number of training epochs.
            n_enc1 (int): The number of encoders in the first stage.
            n_enc2 (int): The number of encoders in the second stage.
            n_head (int): The number of attention heads.
            lr (float): The learning rate.
            dim_ff (int): The dimension of the feed-forward network.
            dropout_ff (float): The dropout rate for the feed-forward network.
            concat_mode (str): The concatenation mode to use in the model.
            select_mode (str): The selection mode to use in the model.
            embedding_mode (str): The embedding mode to use in the model.

        Returns:
            float: The test accuracy of the pre-trained model.
        """
        
        X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, fourie_mode, max_seq_len)
        
        train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, n_epochs)

        test_acc = train.test(model_path, patch_size, n_enc1, n_enc2, n_head,
                                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode)

        return  test_acc             
    
    
    def tuning_param(self, param_name, modes, change_data, additional_param=False):

        """
        Perform hyperparameter tuning for a specified parameter.

        This function iterates over possible values (modes) for a given parameter (param_name) and evaluates
        the model's performance. If the model's performance improves with a new value, the parameter is updated.
        
        Args:
            param_name (str): The name of the parameter to tune.
            modes (list): A list of possible values for the parameter.
            change_data (bool): Whether to change the preprocessing during tuning.
            additional_param (str, optional): An additional parameter to tune along with param_name. Defaults to False.
        
        Returns:
            None
        """

        print(f'Tuning {param_name} ..')
        params = self.params
        params_dict = {}
        for param in params:
            init_param_value = getattr(self, f'init_{param}')
            if param == additional_param:
                params_dict[param] = init_param_value
            else:
                params_dict[param] = getattr(self, param, init_param_value)

        for mode in modes:
            if param_name == 'n_head':
                if self.patch_size % mode != 0 or mode > self.patch_size or mode >= self.limit_n_head:
                    continue
                
            if param_name == 'patch_size':
                if mode <= self.lower_limit_patch_size or mode >= self.upper_limit_patch_size:
                    continue
            
            if param_name in ['n_enc1', 'n_enc2']:
                if mode >= self.limit_n_enc or self.init_n_enc1 >= self.limit_n_enc:
                    continue

            params_dict[param_name] = mode
            
            if change_data:
                X_train, y_train, X_test, y_test = self.preprocess.forward(**params_dict)
                train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, self.max_epochs)
                params_dict['patch_size'] = train.path_sizes[len(train.path_sizes) // 2]
            else:
                train = self.train

            metric, val_epoch, _ = train.forward(**params_dict)

            if metric > self.max_metric or (metric == self.max_metric and val_epoch < self.min_val_epoch):
                setattr(self, param_name, mode)
                self.max_metric = metric
                self.min_val_epoch = val_epoch

                if change_data:
                    self.train = train
                    self.init_patch_size = params_dict['patch_size']

                if additional_param:
                    # если подбираем сразу 2 параметра
                    setattr(self, additional_param, getattr(self, f'init_{additional_param}'))
        
        # если подбор не дал лучшего результата, то присваиваем начальное значение
        if not hasattr(self, param_name):
            setattr(self, param_name, getattr(self, f'init_{param_name}'))

        if param_name not in ['select_mode', 'n_enc2']:
            print(f'Metric = {self.max_metric}\n{param_name} = {getattr(self, param_name)}')
        

    def forward(self):

        """
        Perform the tuning of various parameters and save the best model.

        This function iterates through a predefined order of parameters to be tuned (`self.tuning_order`) and 
        optimizes each one. If the parameter is a tuple of two parameters, it tunes them simultaneously. After 
        tuning, it selects the best epoch and performs the final training with the best parameters, then saves the model.

        Returns:
            model (torch.nn.Module): The final trained model.
            params (dict): The parameters of the final model.
            metric (float): The performance metric of the final model.
        """

        for param_name in self.tuning_order:

            if isinstance(param_name, str):

                if param_name == 'patch_size':
                    params = {'modes': self.train.path_sizes, 'change_data': False}
                else:
                    params = self.tuning_dict[param_name]

                self.tuning_param(param_name, params['modes'], params['change_data'])
            else:
                # подбор одновременно двух параметров
                params1 = self.tuning_dict[param_name[0]]
                params2 = self.tuning_dict[param_name[1]]

                init_value = getattr(self, f'init_{param_name[0]}')

                for param1 in params1['modes']:
                    setattr(self, f'init_{param_name[0]}', param1)
                    self.tuning_param(param_name[1], params2['modes'], params2['change_data'], additional_param=param_name[0])
                
                # если подбор не дал лучшего результата, то присваиваем начальное значение
                if not hasattr(self, param_name[0]):
                    setattr(self, param_name[0], init_value)

                print(f'Metric = {self.max_metric}')
                print(f'{param_name[0]} = {getattr(self, param_name[0])}')
                print(f'{param_name[1]} = {getattr(self, param_name[1])}')
        
        print('Select best epoch..')
        self.train.epochs = self.train.epochs * 2
        metric, val_epoch, _ = self.train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.n_head, self.lr, \
                                            self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
        self.n_epochs = val_epoch + 1
    
        print('Final train..')
        self.train.epochs = self.n_epochs
        metric, val_epoch, model = self.train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.n_head, self.lr, \
                                            self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
        return model, self.get_params(), metric 
        


        


        

        

        

        

        


            
            
            



    
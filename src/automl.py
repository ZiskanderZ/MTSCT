import numpy as np

from src.preprocessing import Preprocess
from src.training import Train
from src.config import seed, val_size, batch_size, tuning_dict

class AutoML:

    def __init__(self, data_train_path, data_test_path) -> None:
        
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path

        self.preprocess = Preprocess(self.data_train_path, self.data_test_path)
        train_size = self.preprocess.X_train.shape
        self.max_epochs = int(self.get_num_epochs(train_size[0] * train_size[1] * train_size[2]))

        self.seed = seed
        self.val_size = val_size
        self.batch_size = batch_size

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


    def get_num_epochs(self, size):

        return 20000000 / size
    

    def get_params(self):

        return {'scale_mode': self.scale_mode, 'fourie_mode': self.fourie_mode, 'concat_mode': self.concat_mode, 
                'select_mode': self.select_mode, 'patch_size': self.patch_size, 'n_enc1': self.n_enc1, 'n_enc2': self.n_enc2, 
                'lr': self.lr, 'dim_ff': self.dim_ff, 'dropout_ff': self.dropout_ff, 'n_head': self.n_head, 
                'embedding_mode': self.embedding_mode, 'n_epochs': self.n_epochs, 'max_seq_len': self.init_max_seq_len}
    
    
    def test(self, scale_mode, fourie_mode, max_seq_len, patch_size, n_epochs, n_enc1, n_enc2, n_head, lr,
                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode):
        
        X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, fourie_mode, max_seq_len)
        
        train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, n_epochs)
        metric, val_epoch, model = train.forward(patch_size, n_enc1, n_enc2, n_head, lr, \
                                            dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode)
        
        return metric
    
    
    def select_param(self, param_name, modes, change_data, additional_param=False):

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
                if self.patch_size % mode != 0 or mode > self.patch_size:
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

        print(self.max_metric, getattr(self, param_name))


    def forward(self):

        for param_name in self.tuning_order:

            if isinstance(param_name, str):

                if param_name == 'patch_size':
                    params = {'modes': self.train.path_sizes, 'change_data': False}
                else:
                    params = self.tuning_dict[param_name]

                self.select_param(param_name, params['modes'], params['change_data'])
            else:
                # подбор одновременно двух параметров
                params1 = self.tuning_dict[param_name[0]]
                params2 = self.tuning_dict[param_name[1]]

                init_value = getattr(self, f'init_{param_name[0]}')

                for param1 in params1['modes']:
                    setattr(self, f'init_{param_name[0]}', param1)
                    self.select_param(param_name[1], params2['modes'], params2['change_data'], additional_param=param_name[0])
                
                # если подбор не дал лучшего результата, то присваиваем начальное значение
                if not hasattr(self, param_name[0]):
                    setattr(self, param_name[0], init_value)

                print(param_name[0], getattr(self, param_name[0]))
                print(param_name[1], getattr(self, param_name[1]))
        
        print('Select best epoch..')
        self.train.epochs = self.train.epochs * 2
        metric, val_epoch, _ = self.train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.n_head, self.lr, \
                                            self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
        self.n_epochs = val_epoch + 1
        print(self.n_epochs)
    
        print('Final train..')
        self.train.epochs = self.n_epochs
        metric, val_epoch, model = self.train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.n_head, self.lr, \
                                            self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
        print(metric, val_epoch)

        return model, self.get_params(), metric 
        


        


        

        

        

        

        


            
            
            



    
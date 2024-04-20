import numpy as np

from src.preprocessing import Preprocess
from src.training import Train
from src.config import seed, val_size, batch_size

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

        self.scale_modes = [0, 1, 2]
        self.fourie_modes = [False, True]
        self.concat_modes = [0, 1, 2, 3]
        self.select_modes = [0, 1, 2, 3]
        self.learning_rates = [0.00001, 0.0001, 0.001, 0.01]
        self.dims_ff = [516, 1024, 2048]
        self.dropouts_ff = [0.1, 0.25, 0.5, 0.75]
        self.ns_enc1 = [0, 1, 2, 3]
        self.ns_enc2 = [0, 1, 2, 3]
        self.ns_head = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.embedding_modes = [False, True]
        
        self.init_need_fourie = False
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


    def get_num_epochs(self, size):

        return 20000000 / size
    

    def get_params(self):

        return {'scale_mode': self.scale_mode, 'need_forie': self.fourie_mode, 'concat_mode': self.concat_mode, 
                'select_mode': self.select_mode, 'patch_size': self.patch_size, 'n_enc1': self.n_enc1, 'n_enc2': self.n_enc2, 
                'lr': self.lr, 'dim_ff': self.dim_ff, 'dropout_ff': self.dropout_ff, 'n_head': self.n_head, 'embedding_mode': self.embedding_mode}
    
    
    def test(self, scale_mode, need_fourie, max_seq_len, patch_size, max_epochs, n_enc1, n_enc2, n_head, lr,
                dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode):
        
        X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, need_fourie, max_seq_len)
        
        train = Train(X_train, y_train, X_test, y_test, seed, val_size, batch_size, max_epochs)
        metric, val_epoch, model = train.forward(patch_size, n_enc1, n_enc2, n_head, lr, \
                                            dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode)
        
        return metric


    def forward(self):

        print('Select scale mode..')
        max_metric, min_val_epoch = 0, np.inf
        for scale_mode in self.scale_modes:

            X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, self.init_need_fourie, self.init_max_seq_len)
        
            train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, self.max_epochs)
            patch_size = train.path_sizes[len(train.path_sizes) // 2]
            metric, val_epoch, _ = train.forward(patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.init_lr, \
                                                self.init_dim_ff, self.init_dropout_ff, self.init_concat_mode, self.init_select_mode, self.init_embedding_mode)
            
            if metric > max_metric:
                self.scale_mode = scale_mode
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.scale_mode = scale_mode
                    min_val_epoch = val_epoch

        print(max_metric, self.scale_mode)
        
        print('Select Fourie mode..')
        max_metric, min_val_epoch = 0, np.inf
        for fourie_mode in self.fourie_modes:

            X_train, y_train, X_test, y_test = self.preprocess.forward(scale_mode, fourie_mode, self.init_max_seq_len)
        
            train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, self.max_epochs)
            patch_size = train.path_sizes[len(train.path_sizes) // 2]
            metric, val_epoch, _ = train.forward(patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.init_lr, \
                                                self.init_dim_ff, self.init_dropout_ff, self.init_concat_mode, self.init_select_mode, self.init_embedding_mode)
            
            if metric > max_metric:
                self.fourie_mode = fourie_mode
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.fourie_mode = fourie_mode
                    min_val_epoch = val_epoch            
        print(max_metric, self.fourie_mode)

        X_train, y_train, X_test, y_test = self.preprocess.forward(self.scale_mode, self.fourie_mode, self.init_max_seq_len)
        train = Train(X_train, y_train, X_test, y_test, self.seed, self.val_size, self.batch_size, self.max_epochs)
        patch_size = train.path_sizes[len(train.path_sizes) // 2]
        
        print('Select concat and select modes..')
        max_metric, min_val_epoch = 0, np.inf
        for concat_mode in self.concat_modes:
            for select_mode in self.select_modes:
                metric, val_epoch, _ = train.forward(patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.init_lr, \
                                                    self.init_dim_ff, self.init_dropout_ff, concat_mode, select_mode, self.init_embedding_mode)
            
                if metric > max_metric:
                    self.concat_mode = concat_mode
                    self.select_mode = select_mode
                    max_metric = metric
                    min_val_epoch = val_epoch
                elif metric == max_metric:
                    if val_epoch < min_val_epoch:
                        self.concat_mode = concat_mode
                        self.select_mode = select_mode
                        min_val_epoch = val_epoch 
                    
        print(max_metric, self.concat_mode, self.select_mode)
        
        print('Select patch size..')
        max_metric, min_val_epoch = 0, np.inf
        for patch_size in train.path_sizes:
        
            metric, val_epoch, _ = train.forward(patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.init_lr, \
                                                    self.init_dim_ff, self.init_dropout_ff, self.concat_mode, self.select_mode, self.init_embedding_mode)
        
            if metric > max_metric:
                self.patch_size = patch_size
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.patch_size = patch_size
                    min_val_epoch = val_epoch  
        print(max_metric, self.patch_size)

        print('Select learning rate..')
        max_metric, min_val_epoch = 0, np.inf
        for lr in self.learning_rates:
        
            metric, val_epoch, _ = train.forward(self.patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, lr, \
                                                    self.init_dim_ff, self.init_dropout_ff, self.concat_mode, self.select_mode, self.init_embedding_mode)
        
            if metric > max_metric:
                self.lr = lr
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.lr = lr
                    min_val_epoch = val_epoch  
        print(max_metric, self.lr)
        
        print('Select feed forward neurons..')
        max_metric, min_val_epoch = 0, np.inf
        for dim_ff in self.dims_ff:
        
            metric, val_epoch, _ = train.forward(self.patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.lr, \
                                                dim_ff, self.init_dropout_ff, self.concat_mode, self.select_mode, self.init_embedding_mode)
        
            if metric > max_metric:
                self.dim_ff = dim_ff
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.dim_ff = dim_ff
                    min_val_epoch = val_epoch  
        print(max_metric, self.dim_ff)

        print('Select feed forward dropout..')
        max_metric, min_val_epoch = 0, np.inf
        for dropout_ff in self.dropouts_ff:
        
            metric, val_epoch, _ = train.forward(self.patch_size, self.init_n_enc1, self.init_n_enc2, self.init_n_head, self.lr, \
                                                self.dim_ff, dropout_ff, self.concat_mode, self.select_mode, self.init_embedding_mode)
        
            if metric > max_metric:
                self.dropout_ff = dropout_ff
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.dropout_ff = dropout_ff
                    min_val_epoch = val_epoch 
        print(max_metric, self.dropout_ff)

        print('Select nums encoders..')
        max_metric, min_val_epoch = 0, np.inf
        for n_enc1 in self.ns_enc1:
            for n_enc2 in self.ns_enc2:
        
                metric, val_epoch, _ = train.forward(self.patch_size, n_enc1, n_enc2, self.init_n_head, self.lr, \
                                                    self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.init_embedding_mode)

                if metric > max_metric:
                    self.n_enc1 = n_enc1
                    self.n_enc2 = n_enc2
                    max_metric = metric
                    min_val_epoch = val_epoch
                elif metric == max_metric:
                    if val_epoch < min_val_epoch:
                        self.n_enc1 = n_enc1
                        self.n_enc2 = n_enc2
                        min_val_epoch = val_epoch 
        print(max_metric, self.n_enc1, self.n_enc2)

        print('Select need embedding layer..')
        max_metric, min_val_epoch = 0, np.inf
        for embedding_mode in self.embedding_modes:
        
            metric, val_epoch, _ = train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.init_n_head, self.lr, \
                                                    self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, embedding_mode)
        
            if metric > max_metric:
                self.embedding_mode = embedding_mode
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.embedding_mode = embedding_mode
                    min_val_epoch = val_epoch  
        print(max_metric, self.embedding_mode)

        print('Select num attention heads..')
        max_metric, min_val_epoch = 0, np.inf
        for n_head in self.ns_head:
            if self.patch_size % n_head != 0 or n_head > self.patch_size:
                continue

            metric, val_epoch, _ = train.forward(self.patch_size, self.n_enc1, self.n_enc2, n_head, self.lr, \
                                                self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
            if metric > max_metric:
                self.n_head = n_head
                max_metric = metric
                min_val_epoch = val_epoch
            elif metric == max_metric:
                if val_epoch < min_val_epoch:
                    self.n_head = n_head
                    min_val_epoch = val_epoch 
        print(max_metric, self.n_head)
        
        print('Final train..')
        train.epochs = train.epochs * 2
        metric, val_epoch, model = train.forward(self.patch_size, self.n_enc1, self.n_enc2, self.n_head, self.lr, \
                                            self.dim_ff, self.dropout_ff, self.concat_mode, self.select_mode, self.embedding_mode)
        
        
        return metric, self.get_params()
        


        


        

        

        

        

        


            
            
            



    
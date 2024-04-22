seed = 23
val_size = 0.1
batch_size = 16

tuning_dict = {'scale_mode': {'modes': [0, 1, 2], 'change_data': True},
                'fourie_mode': {'modes': [False, True], 'change_data': True},
                'concat_mode': {'modes': [0, 1, 2, 3], 'change_data': False},
                'select_mode': {'modes': [0, 1, 2, 3], 'change_data': False},
                'lr': {'modes': [0.00001, 0.0001, 0.001, 0.01], 'change_data': False},
                'dim_ff': {'modes': [516, 1024, 2048], 'change_data': False},
                'dropout_ff': {'modes': [0.1, 0.25, 0.5, 0.75], 'change_data': False},
                'n_enc1': {'modes': [0, 1, 2, 3], 'change_data': False},
                'n_enc2': {'modes': [0, 1, 2, 3], 'change_data': False},
                'n_head': {'modes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'change_data': False},
                'embedding_mode': {'modes': [False, True], 'change_data': False},
                }
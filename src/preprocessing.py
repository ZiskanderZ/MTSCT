import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.io.arff import loadarff 

class Preprocess:

    def __init__(self, data_train_path, data_test_path) -> None:
        
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path

        self.load_data()
    
    def load_data(self):

        print('Data loading..')

        data_train = loadarff(self.data_train_path)
        data_test = loadarff(self.data_test_path)

        self.X_train, self.y_train = self.get_data(data_train)
        self.X_test, self.y_test = self.get_data(data_test)

    def scale(self, X, mode):

        if mode == 0:
            max_shape = len(X.shape) - 1
            min_val = np.min(X, axis=max_shape, keepdims=True)
            max_val = np.max(X, axis=max_shape, keepdims=True)

            return (X - min_val)  / (max_val - min_val + 0.00000000001)
        
        elif mode == 1:
            min_val = X.min()
            max_val = X.max()

            return (X - min_val)  / (max_val - min_val + 0.00000000001)
        
        elif mode == 2:

            return X
        
    def get_padded_data(self, data, max_seq_len, fourie, scale_mode):

        if pd.isna(max_seq_len):
            max_seq_len = max(list(map(lambda x: x.shape[1], data)))

        data_padded = np.zeros((len(data), len(data[0]), max_seq_len))
        for num, arr in enumerate(data):
            if fourie:
                arr = np.abs(np.fft.fft(arr, axis=1))
            arr = self.scale(arr, scale_mode)
            arr = np.pad(arr, ((0, 0), (0, max_seq_len-arr.shape[1])), constant_values=-1)
            data_padded[num] = arr
        
        return data_padded

    def get_data(self, data): 

        X, y = [], []

        for i in tqdm(data[0]):
            features, label = i

            shape_arr = np.array(list(features[0]))
            shape = shape_arr[~np.isnan(shape_arr)].shape[0]
            features_arr = np.zeros((features.shape[0], shape))

            for num in range(features.shape[0]):
                feature = np.array(list(features[num]))
                features_arr[num] = feature[~np.isnan(feature)]

            X.append(np.array(features_arr))
            y.append(str(label))

        try:
            X, y = np.array(X), np.array(y)
        except:
            pass

        return X, y

    def forward(self, scale_mode, fourie=False, max_seq_len=None):

        if isinstance(self.X_train, list):
            print('Add padding')
            X_train_padded = self.get_padded_data(self.X_train, max_seq_len, fourie, scale_mode)
            X_test_padded = self.get_padded_data(self.X_test, max_seq_len, fourie, scale_mode)

            if pd.isna(max_seq_len):
                train_shape = X_train_padded.shape[2]
                test_shape = X_test_padded.shape[2]
                if train_shape > test_shape:
                    X_test_padded = self.get_padded_data(self.X_test, train_shape, fourie, scale_mode)
                elif train_shape < test_shape:
                    X_train_padded = self.get_padded_data(self.X_train, test_shape, fourie, scale_mode)
                
            self.X_train = X_train_padded
            self.X_test = X_test_padded

        else:
            if fourie:
                self.X_train = np.abs(np.fft.fft(self.X_train, axis=2))
                self.X_test = np.abs(np.fft.fft(self.X_test, axis=2))

            self.X_train = self.scale(self.X_train, scale_mode)
            self.X_test = self.scale(self.X_test, scale_mode)
        
        return self.X_train, self.y_train, self.X_test, self.y_test
        


    


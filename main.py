from src.automl import AutoML

if __name__ == '__main__':

    ds_name = ...

    train_path = rf'data\{ds_name}\{ds_name}_TRAIN.arff'
    test_path = rf'data\{ds_name}\{ds_name}_TEST.arff'

    automl = AutoML(train_path, test_path)
    model, params, metric = automl.forward()
    metric = automl.test(**params)
    print(metric)
    print(params)
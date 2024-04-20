from src.automl import AutoML

if __name__ == '__main__':

    ds_name = 'AtrialFibrillation'

    train_path = rf'data\{ds_name}\{ds_name}_TRAIN.arff'
    test_path = rf'data\{ds_name}\{ds_name}_TEST.arff'

    automl = AutoML(train_path, test_path)
    metric, params = automl.forward()
    # metric = automl.test(0, False, None, 20, 2000, 1, 3, 1, 0.0001, 2048, 0.1, 0, 2, False)
    print(metric)
    print(params)
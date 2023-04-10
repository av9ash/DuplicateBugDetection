import os
import json
from collections import OrderedDict
from baseline_ir.nn_model import NNModel
from tqdm import tqdm


def load_pr_data(data_path):
    '''
    Creates json object from description and parameters files.
    :param data_path
    :return: dictionary
    '''
    pr_data = OrderedDict()
    files_to_load = os.listdir(data_path)

    if '.DS_Store' in files_to_load:
        files_to_load.remove('.DS_Store')

    files_to_load = sorted(files_to_load, key=lambda i: int(i), reverse=True)
    for i, pr_number in enumerate(files_to_load):
        description_file = data_path + '/' + pr_number + '/' + pr_number + '_description.txt'
        properties_file = data_path + '/' + pr_number + '/' + pr_number + '_properties.json'
        if os.path.exists(description_file):
            with open(description_file) as des_file_obj:
                description = des_file_obj.read()

        if os.path.exists(properties_file):
            with open(properties_file) as prop_file_obj:
                pr_features = json.load(prop_file_obj)
                pr_features['product'] = pr_features.get('product', '').strip('\n')
                pr_features['pr_description'] = description
                pr_features.pop('submitter-id')
                pr_data[pr_features.pop('number')] = pr_features

    return pr_data


def test_model(X_test, test_y, n_neighbors, model):
    print('Testing Model..')
    y_preds = []
    pbar = tqdm(total=len(test_y))
    for i, item in enumerate(X_test):
        similar_prs = model.predict(item, n_neighbors)
        y_preds.append(similar_prs)
        pbar.update(1)
    pbar.close()
    print('Done')


def main(train_path, test_path):
    is_test = 'True'
    is_train = 'True'
    n_neighbors = 5
    model = NNModel()

    if is_train == 'True':
        pr_data = load_pr_data(train_path)
        train_X, train_y = model.get_X_y(pr_data)
        model.fit(train_X, train_y)
        model.save()

    if is_test == 'True':
        pr_data = load_pr_data(test_path)
        test_X, test_y = model.get_X_y(pr_data)
        model.load()
        X_test = model.transform(test_X)
        test_model(X_test, test_y, n_neighbors, model)


if __name__ == '__main__':
    train_path = '/Users/patila/Desktop/gnats_data/21fq_quick/training'
    test_path = '/Users/patila/Desktop/gnats_data/21fq_quick/testing'
    main(train_path, test_path)

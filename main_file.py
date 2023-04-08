import os
import json
from collections import OrderedDict


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


def main(train_path, test_path):
    is_test = 'True'
    is_train = 'True'
    n_neighbors = 5
    model_name = 'ir_knn_model'

    if is_train == 'True':
        pr_data = load_pr_data(train_path)

    if is_test == 'True':
        pr_data = load_pr_data(test_path)


if __name__ == '__main__':
    train_path = '/Users/patila/Desktop/gnats_data/21fq_quick/training'
    test_path = '/Users/patila/Desktop/gnats_data/21fq_quick/testing'
    main(train_path, test_path)

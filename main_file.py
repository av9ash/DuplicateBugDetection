import os
import json
from collections import OrderedDict
from baseline_ir.nn_model import NNModel
# from fasttext_ir.nn_ft_model import FTNNModel
from gensim_ir.d2v_model import GenModel
from tqdm import tqdm
from print_plots import scatter_plot, histo_sim, top5_dist


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


def get_dup_org_maps(data_dir):
    with open(data_dir + '/dup_org_map.json') as f:
        duo_map = json.load(f)

    return duo_map


def create_results_dir(data_dir, model_dir, model_name):
    bugs_repo = data_dir.split('/')[-1]
    repo_path = os.path.join(model_dir, bugs_repo)
    if not os.path.exists(repo_path):
        os.mkdir(repo_path)
    plots_path = os.path.join(repo_path, model_name)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    return plots_path


def get_similar_prs(X_test, test_y, n_neighbors, model):
    print('Generating Recommendations..')
    recommendations = {}
    y_preds = []
    if type(X_test) is list:
        pbar_tot = len(X_test)
    else:
        pbar_tot = X_test.shape[0]

    pbar = tqdm(total=pbar_tot)
    for i, item in enumerate(X_test):
        query_pr = test_y[i]
        similar_prs = model.predict(item, n_neighbors)
        y_preds.append(similar_prs)
        recommendations[query_pr] = similar_prs
        pbar.update(1)
    pbar.close()
    print('Done')
    return y_preds, recommendations


def evaluate_model(test_prs, y_preds, duo_map):
    # Check recommendations against original and rest of the marked duplicate prs as a set
    pos_sim = []
    print('Evaluating Model..')
    count = 0
    for i, dup_pr in enumerate(test_prs):
        # if duo2_map.get(dup_pr, '') in y_pred[i]:
        # Following is if recommenation is not original but any other known duplicates,
        org = duo_map.get(dup_pr, '')
        # check if the sets have an intersection
        if org in y_preds[i]:
            count += 1
            # print('Potential Dup: {}'.format(dup_pr), 'Similar: {}'.format(y_pred[i]), 'Found: {}'.format(count))
            pos = list(y_preds[i].keys()).index(org)
            sim = y_preds[i].get(org)
            pos_sim.append({'sim': sim, 'pos': pos})
        else:
            sim = list(y_preds[i].values())[0]
            pos_sim.append({'sim': sim, 'pos': -1})

    acc = round(count / len(test_prs), 2)
    print('Accuracy: {}%'.format(acc))
    # print(pos_sim)
    return pos_sim


def main(train_path, test_path, n_neighbors, model, model_name):
    is_test = 'True'
    is_train = 'True'
    print_plots = 'True'

    data_dir = os.path.dirname(train_path)
    plots_path = create_results_dir(data_dir, model.model_dir, model_name)

    if is_train == 'True':
        print('Training..')
        pr_data = load_pr_data(train_path)
        train_X, train_y = model.get_X_y(pr_data)
        model.fit(train_X, train_y)
        model.save()

    if is_test == 'True':
        print('Testing..')
        pr_data = load_pr_data(test_path)
        test_X, test_y = model.get_X_y(pr_data)
        model.load()
        X_test = model.transform(test_X)
        y_preds, recs = get_similar_prs(X_test, test_y, n_neighbors, model)

        duo_map = get_dup_org_maps(data_dir)
        pos_sim = evaluate_model(test_y, y_preds, duo_map)
        with open(plots_path + '/pos_sim.json', 'w') as f:
            json.dump(pos_sim, f)

        with open(plots_path + '/recs.json', 'w') as f:
            json.dump(recs, f)

    if print_plots == 'True':
        with open(plots_path + '/pos_sim.json') as f:
            pos_sim = json.load(f)

        scatter_plot(pos_sim, plots_path)
        histo_sim(pos_sim, plots_path)
        top5_dist(pos_sim, plots_path)


if __name__ == '__main__':
    model1 = NNModel()
    model2 = GenModel()
    # model3 = FTNNModel()

    # train_path = '/Users/patila/Desktop/gnats_data/21fq_quick/training'
    # test_path = '/Users/patila/Desktop/gnats_data/21fq_quick/testing'

    # repo = 'Thunderbird'
    # train_path = '/Users/patila/Desktop/open_data/bugrepo/{}/training'.format(repo)
    # test_path = '/Users/patila/Desktop/open_data/bugrepo/{}/testing'.format(repo)

    # model_name = 'all_data'
    # n_neighbors = 5
    # main(train_path, test_path, n_neighbors, model1, model_name)

    models = [model1, model2]
    repos = ['Thunderbird', 'JDT', 'EclipsePlatform', 'Firefox', 'MozillaCore']
    open_data_path = '/Users/patila/Desktop/open_data/bugrepo'
    for repo in repos:
        for model in models:
            print('Bugs Repo: ', repo)
            print('Model: ', type(model))
            train_path = '{}/{}/training'.format(open_data_path, repo)
            test_path = '{}/{}/testing'.format(open_data_path, repo)
            model_name = 'all_data'

            n_neighbors = 5
            main(train_path, test_path, n_neighbors, model, model_name)

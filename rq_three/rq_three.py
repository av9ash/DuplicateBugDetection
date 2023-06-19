import json
from datetime import datetime, timedelta
import joblib
import numpy
from sklearn.neighbors import NearestNeighbors
import time
from scipy.spatial import distance
import numpy as np
from main_file import load_pr_data
from baseline_ir.nn_model import NNModel
# from bert_ir.bert_base import BERT_NNModel
# from fasttext_ir.nn_ft_model import FTNNModel
# from gensim_ir.d2v_model import GenModel
from main_file import *
from scipy.sparse import vstack
from tqdm import tqdm


def load_ct_prs_mapping(model_class_name, model_name):
    ct_prs_mapping = {}
    prct_file_name = 'embeddings/{}_{}.json'.format(model_class_name, model_name)

    with open(prct_file_name, 'r') as f:
        pr_ct_map = json.load(f)

    date_format = "%Y-%m-%d %H:%M:%S %z"
    for pr in sorted(pr_ct_map.keys()):
        cdt = datetime.strptime(pr_ct_map[pr], date_format)
        cdt_date = cdt.date()
        date_str = cdt_date.strftime("%Y-%m-%d")
        if date_str in ct_prs_mapping:
            ct_prs_mapping[date_str].append(pr)
        else:
            ct_prs_mapping[date_str] = [pr]

    # Sort the creation date to pr list mapping by keys.
    res = {}
    for key in sorted(ct_prs_mapping.keys()):
        res[key] = ct_prs_mapping[key]

    return res


def get_pr_ctime_map(pr_data):
    prct_map = {}
    for pr in sorted(pr_data.keys()):
        if 'creation_time' in pr_data[pr]:
            ct = pr_data[pr]['creation_time']
        else:
            ct = pr_data[pr]['arrival-date']

        prct_map[pr] = ct

    return prct_map


def get_closest_date(input_date, all_dates):
    date_format = "%Y-%m-%d"
    input_datetime_date = datetime.strptime(input_date, date_format)
    all_datetime_dates = [datetime.strptime(x, date_format) for x in all_dates]
    input_all_delta = [x - input_datetime_date for x in all_datetime_dates]
    min_val = min(filter(lambda x: x.days >= 0, input_all_delta))
    min_pos = input_all_delta.index(min_val)
    # print(input_date, all_dates[min_pos])
    return all_dates[min_pos]


def get_prs_to_search(start_date, end_date, ct_prs_map):
    prs = []
    all_dates = list(ct_prs_map.keys())

    # If the exact start or end date is not find get the next closest.
    if start_date not in all_dates:
        # print('Start Date Replaced:')
        start_date = get_closest_date(start_date, all_dates)

    if end_date not in all_dates:
        # print('End Date Replaced:')
        end_date = get_closest_date(end_date, all_dates)

    start_idx = all_dates.index(start_date)
    end_idx = all_dates.index(end_date)
    # print(start_idx, end_idx)

    if end_idx < len(all_dates) - 1:
        end_idx += 1

    search_period = all_dates[start_idx:end_idx]
    for date in search_period:
        prs.extend(ct_prs_map[date])

    return prs


def get_pr_and_embedding(search_prs, pr_emb_map):
    res = {}
    for pr in search_prs:
        if type(pr_emb_map[pr]) is numpy.ndarray:
            res[pr] = pr_emb_map[pr]
        else:
            res[pr] = pr_emb_map[pr].toarray().ravel()
            # Test for NN Model without ravel
            # res[pr] = pr_emb_map[pr]
    return res


def get_model(X_train):
    nn = NearestNeighbors(n_jobs=-1, p=2, algorithm='auto')
    nn.fit(X_train)
    return nn


def get_test_embeddings(pr_data, model):
    print('Tranforming..')
    corpus, labels = model.get_X_y(pr_data)
    embeddings = model.transform(corpus)
    # Save embeddings
    return embeddings, labels


def compute_distances(X_train, x_test):
    return distance.cdist(x_test, X_train, metric='euclidean')


def predict(x_test, top_n, X_train):
    distances = compute_distances(X_train, x_test)

    index_mins = np.argsort(distances[0])[:top_n]
    dkt = {}
    for idx in index_mins:
        dkt[idx] = distances[0][idx]
    return dkt


def get_duplicates(X_train, Y_train, X_test, top_n=5):
    idx_n_dists = predict(X_test, top_n, X_train)
    prnum_n_sim = {}
    for indx, distance in idx_n_dists.items():
        pr = Y_train[indx]
        prnum_n_sim[pr] = round((1 - distance) * 100, 2)

    return prnum_n_sim


def get_search_cluster(ct, ct_prs_map, search_days):
    date_format = "%Y-%m-%d %H:%M:%S %z"
    creation_date = datetime.strptime(ct, date_format)
    lookback_date = (creation_date - timedelta(days=search_days))
    end_date = creation_date.strftime('%Y-%m-%d')
    start_date = lookback_date.strftime('%Y-%m-%d')
    search_prs = get_prs_to_search(start_date, end_date, ct_prs_map)
    return search_prs


def exp_three(train_path, test_path, n_neighbors, model, model_name, search_days, is_train='False'):
    is_test = 'True'
    accuracy = {}
    data_dir = os.path.dirname(train_path)
    emb_file_name = 'embeddings/{}_{}.joblib'.format(model.__class__.__name__, model_name)
    plots_path = create_results_dir(model.model_dir, data_dir, model_name)
    cluster_sizes = []

    if is_train == 'True':
        print('Training..')
        pr_data = load_pr_data(train_path)
        pr_ct_map = get_pr_ctime_map(pr_data)
        train_X, train_y = model.get_X_y(pr_data)
        model.fit(train_X, train_y)

        # Save Embeddings
        joblib.dump(model.embeddings, emb_file_name)
        # Save PR to Creation Time mappings.
        prct_file_name = 'embeddings/{}_{}.json'.format(model.__class__.__name__, model_name)
        with open(prct_file_name, 'w') as f:
            json.dump(pr_ct_map, f)

        # Save Model, this removes embeddings so call this after saving them.
        model.save()

    if is_test == 'True':
        print('Testing..')
        pr_data = load_pr_data(test_path)
        ct_prs_map = load_ct_prs_mapping(model.__class__.__name__, model_name)
        model.load()
        model.embeddings = joblib.load(emb_file_name)
        test_X, test_y = model.get_X_y(pr_data)
        # test_X = test_X[:1200]
        # test_y = test_y[:1200]
        X_test = model.transform(test_X)
        joblib.dump(X_test, 'embeddings/{}_{}_test.joblib'.format(model.__class__.__name__, model_name))

        pr_test_embs = dict(zip(test_y, X_test))
        y_preds = []
        recommendations = {}

        print('\nGenerating Recommendations.. ')
        pbar = tqdm(total=len(pr_data.keys()))
        for pr in sorted(pr_data.keys()):
            if 'creation_time' in pr_data[pr]:
                ct = pr_data[pr]['creation_time']
            else:
                ct = pr_data[pr]['arrival-date']

            search_prs = get_search_cluster(ct, ct_prs_map, search_days)
            cluster_sizes.append(len(search_prs))

            pr_embeddings_map = get_pr_and_embedding(search_prs, model.embeddings)
            X_train = list(pr_embeddings_map.values())
            train_y = list(pr_embeddings_map.keys())
            x_test = [pr_test_embs[pr].toarray().ravel()]

            # # Changes for NN model:
            # if type(pr_test_embs[pr]) is numpy.ndarray:
            #     x_test = pr_test_embs[pr]
            # else:
            #     x_test = pr_test_embs[pr].toarray().ravel()
            # search_model = get_model(X_train)

            similar_prs = get_duplicates(X_train, train_y, [x_test])
            y_preds.append(similar_prs)
            recommendations[pr] = similar_prs
            pbar.update()

        pbar.close()

        with open(plots_path + '/recs.json', 'w') as f:
            json.dump(recommendations, f)

        duo_map = get_dup_org_maps(data_dir)
        pos_sim, acc = evaluate_model(test_y, y_preds, duo_map)
        with open(plots_path + '/pos_sim.json', 'w') as f:
            json.dump(pos_sim, f)

        with open(plots_path + '/pr_searched.json', 'w') as f:
            json.dump(cluster_sizes, f)

    # if print_plots == 'True':
    #     with open(plots_path + '/pos_sim.json') as f:
    #         pos_sim = json.load(f)
    #
    #     scatter_plot(pos_sim, plots_path)
    #     histo_sim(pos_sim, plots_path)
    #     top5_dist(pos_sim, plots_path)

    return accuracy


def run_full_exp():
    # repos = ['Thunderbird', 'JDT', 'EclipsePlatform', 'Firefox', 'MozillaCore']
    open_data_path = '/Users/patila/Desktop/open_data/bugrepo'
    models = [NNModel()]
    repos = ['Thunderbird']
    search_days = {'Thunderbird': 1440, 'JDT': 720, 'EclipsePlatform': 720, 'Firefox': 1080, 'MozillaCore': 900}

    for model in models:
        for repo in repos:
            print('Bugs Repo: ', repo)
            print('Model: ', type(model))
            train_path = '{}/{}/training'.format(open_data_path, repo)
            test_path = '{}/{}/testing'.format(open_data_path, repo)
            n_neighbors = 5
            model_name = '{}_chunks'.format(repo)
            exp_three(train_path, test_path, n_neighbors, model, model_name, search_days[repo], is_train='True')


run_full_exp()

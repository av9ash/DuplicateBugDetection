import json
from main_file import *
from itertools import islice
import matplotlib.pylab as plt
from bert_tsdae_ir.bert_base import BERT_NNModel

# Get change in accuracy with respect to various values of k
def exp_one(train_path, test_path, n_neighbors, model, model_name, is_train='False'):
    is_test = 'True'
    accuracy = {}
    data_dir = os.path.dirname(train_path)
    plots_path = create_results_dir(model.model_dir, data_dir, model_name)

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
        # test_X = test_X[:1200]
        # test_y = test_y[:1200]
        model.load()
        X_test = model.transform(test_X)
        y_preds, recs = get_similar_prs(X_test, test_y, max(n_neighbors), model)

        duo_map = get_dup_org_maps(data_dir)

        for n in n_neighbors:
            # get first n pr nums and sim score from each dict in y_preds.
            y_preds_n = [dict((islice(y.items(), n))) for y in y_preds]
            pos_sim, acc = evaluate_model(test_y, y_preds_n, duo_map)
            accuracy[n] = round(acc, 2)

    return accuracy


def plot_chart(accs, chart_name):
    fig, ax = plt.subplots()
    ax.set_title('Accuracy vs Top N : {}'.format(chart_name))
    ax.set_xlabel('Value of N ->')
    ax.set_ylabel('Accuracy->')
    # plt.show()
    plt.plot(accs.keys(), accs.values())
    plt.savefig(chart_name+'.png')


def run_test_exp():
    train_path = '/Users/patila/Desktop/gnats_data/21_Q1/training'
    test_path = '/Users/patila/Desktop/gnats_data/21_Q1/testing'
    model = NNModel()
    repo = 'gnats'
    model_name = 'all_data'
    n_neighbors = [1]
    n_neighbors += list(range(5, 501, 5))
    accs = exp_one(train_path, test_path, n_neighbors, model, model_name)
    print(accs)
    chart_name = '{}_{}_{}.png'.format(model.__class__.__name__, repo, model_name)
    plot_chart(accs, chart_name)
    print('Chart plotted at: {}'.format(chart_name))


def run_full_exp():
    models = [BERT_NNModel()]
    # repos = ['Firefox', 'MozillaCore']
    # repos = ['Thunderbird', 'JDT', 'EclipsePlatform', 'Firefox', 'MozillaCore']
    repos = ['EclipsePlatform', 'Firefox', 'MozillaCore']
    open_data_path = '/Users/patila/Desktop/open_data/bugrepo'
    for model in models:
        for repo in repos:
            print('Bugs Repo: ', repo)
            print('Model: ', type(model))
            train_path = '{}/{}/training'.format(open_data_path, repo)
            test_path = '{}/{}/testing'.format(open_data_path, repo)
            model_name = 'all_data'

            n_neighbors = [1]
            n_neighbors += list(range(5, 501, 5))
            accs = exp_one(train_path, test_path, n_neighbors, model, model_name, 'True')
            chart_name = '{}_{}'.format(model.__class__.__name__, repo)
            with open(chart_name + '.json', 'w') as f:
                json.dump(accs, f)
            plot_chart(accs, chart_name)


# run_test_exp()
run_full_exp()

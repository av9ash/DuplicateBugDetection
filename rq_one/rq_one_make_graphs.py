import os
import json
import matplotlib.pylab as plt


def get_results_from_files(repo_name):
    result_files = os.listdir('results')
    repo_files = [x for x in sorted(result_files) if repo_name + '.json' in x]
    results = {}
    for r_file in repo_files:
        with open('results/' + r_file, 'r') as f:
            key = r_file.split('.json')[0]
            accuracies = json.load(f)
            results[key] = list(accuracies.values())

    return results


def plot_combined_chart(results, chart_name):
    emb_lbls = {'NNModel': 'TFIDF', 'GenModel': 'Gensim', 'BERT': 'BERT', 'FTNNModel': 'Fasttext'}
    colors = ['Green', 'Red', 'Blue', 'Orange']
    title = chart_name.split('_combined')[0]

    fig, ax = plt.subplots()

    ax.set_title('Accuracy vs Top N : {}'.format(title))
    ax.set_xlabel('Value of N ->')
    ax.set_ylabel('Accuracy->')

    n_neighbors = [1]
    n_neighbors += list(range(5, 501, 5))

    labels = [emb_lbls[x.split('_')[0]] for x in results.keys()]
    for i, accs in enumerate(results.values()):
        plt.plot(n_neighbors, accs, color=colors[i], label=labels[i])

    plt.legend()
    # plt.show()
    if not os.path.exists('charts'):
        os.mkdir('charts')
    plt.savefig('charts/' + chart_name + '.png')


repo_names = ['Thunderbird', 'JDT', 'EclipsePlatform', 'Firefox', 'MozillaCore']

for repo in repo_names:
    results = get_results_from_files(repo)
    plot_combined_chart(results, '{}_combined'.format(repo))
print('Done')

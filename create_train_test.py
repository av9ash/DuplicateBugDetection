import csv
from create_bugs_maps import get_dup_org_maps
import json
import os
import sys
from tqdm import tqdm


def read_csv(target_file):
    """
    Read rows from csv file.
    :param target_file: path where file is saved
    :return: list
    """
    data = {}
    with open(target_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['Issue_id']] = {'number': row['Issue_id'], 'synopsis': row['Title'], 'symptom': '',
                                     'pr-impact': '', 'problem-level': row['Priority'], 'category': row['Component'],
                                     'product': '', 'platform': '', 'submitter-id': '',
                                     'creation_time': row['Created_time'],
                                     'pr_description': row['Description']}
    return data


def dump_pr(pr_details, path):
    pr_details = dict(pr_details.items())
    pr_num = pr_details['number']
    pr_path = os.path.join(path, pr_num)
    if not os.path.exists(pr_path):
        os.mkdir(pr_path)
    pr_details_path = os.path.join(pr_path, pr_num + '_properties.json')
    pr_desc_path = os.path.join(pr_path, pr_num + '_description.txt')

    with open(pr_desc_path, 'w') as f:
        f.write(pr_details['pr_description'])

    with open(pr_details_path, 'w') as f:
        json.dump(pr_details, f)

    return pr_path


def get_train_test(path):
    data_dir = os.path.dirname(path)
    dup_org_map = get_dup_org_maps(data_dir)
    print('Dup PRs', len(set(dup_org_map.keys())))
    print('Org PRs', len(set(dup_org_map.values())))
    data = read_csv(path)
    # data = get_pr_details(rows)
    test_path = os.path.join(data_dir, 'testing')
    train_path = os.path.join(data_dir, 'training')

    if not os.path.exists(test_path):
        os.mkdir(test_path)
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    pbar = tqdm(total=len(dup_org_map))
    for dup, org in dup_org_map.items():
        dump_pr(data[dup], test_path)
        dump_pr(data[org], train_path)
        pbar.update(1)

    pbar.close()
    print('Saved:{} Child Bugs.'.format(len(os.listdir(test_path))))
    print('Saved:{} Parent Bugs.'.format(len(os.listdir(train_path))))


def get_unique_prs(path):
    data_dir = os.path.dirname(path)
    dup_org_map = get_dup_org_maps(data_dir)
    dup = dup_org_map.keys()
    org = dup_org_map.values()

    train_path = os.path.join(data_dir, 'training')
    data = read_csv(path)
    count = []
    pbar = tqdm(total=len(data))
    for pr_num, details in data.items():
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if pr_num not in dup and pr_num not in org:
            count.append(dump_pr(details, train_path))
        pbar.update(1)
    pbar.close()
    # print('Saved:{} Unique Bugs.', len(os.listdir(train_path)))
    print('Saved:{} Unique Bugs.', len(count))


if __name__ == '__main__':
    main_path = '/Users/patila/Desktop/open_data/bugrepo/'
    # main_path = sys.argv[1]
    repos = ['Thunderbird/mozilla_thunderbird.csv', 'JDT/eclipse_jdt.csv', 'EclipsePlatform/eclipse_platform.csv',
             'Firefox/mozilla_firefox.csv', 'MozillaCore/mozilla_core.csv']

    for repo in repos:
        print(repo)
        csv_path = os.path.join(main_path, repo)
        get_train_test(csv_path)
        get_unique_prs(csv_path)

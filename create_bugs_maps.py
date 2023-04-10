import csv
import json
import os


def read_csv(target_file):
    """
    Read rows from csv file.
    :param target_file: path where file is saved
    :return: list
    """
    rows = {}
    with open(target_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows[row['Issue_id']] = row['Duplicate']
    return rows


def make_dup_org_mapppings(target_file):
    data = read_csv(target_file)
    dup_org = {}
    # all_duplicates = get_all_duplicate_prs(data)
    for org, dups in data.items():
        if org not in dup_org and dups != 'NULL':
            duplicates = dups.split(';')
            for duplicate in duplicates:
                if duplicate not in dup_org:
                    dup_org[duplicate] = org
                else:
                    # If and org ends up here that simply means that it was linked to a
                    # sibling report as original instead of being linked to a parent as duplicate.
                    # dup_org[duplicate].append(org)
                    sibling_dup = org
                    org = dup_org[duplicate]
                    dup_org[sibling_dup] = org

    return dup_org


def get_dup_org_maps(bug_repo_path):
    train_dup_org = make_dup_org_mapppings(bug_repo_path + '/train.csv')
    test_dup_org = make_dup_org_mapppings(bug_repo_path + '/test.csv')

    dup_org_merge = {}

    # Add Train PR Maps if they are not common in Train and Test
    for tr_dup, tr_org in train_dup_org.items():
        if tr_dup not in test_dup_org:
            dup_org_merge[tr_dup] = tr_org

    # Add Test PR Maps and Remaining Common PR maps with correct linking to parent
    for ts_dup, ts_org in test_dup_org.items():
        if ts_dup not in train_dup_org:
            dup_org_merge[ts_dup] = ts_org
        else:
            dup_org_merge[ts_dup] = ts_org
            tr_org = test_dup_org[ts_dup]
            dup_org_merge[tr_org] = ts_org

    with open(bug_repo_path + '/dup_org_map.json', 'w') as f:
        json.dump(dup_org_merge, f)

    # print('Done')
    return dup_org_merge


# get_dup_org_maps('/Users/patila/Desktop/open_data/bugrepo/Thunderbird')
# get_dup_org_maps('/Users/patila/Desktop/open_data/bugrepo/MozillaCore')
# get_dup_org_maps('/Users/patila/Desktop/open_data/bugrepo/JDT')
# get_dup_org_maps('/Users/patila/Desktop/open_data/bugrepo/Firefox')
# get_dup_org_maps('/Users/patila/Desktop/open_data/bugrepo/EclipsePlatform')

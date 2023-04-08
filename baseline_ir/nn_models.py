import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial import distance
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.pipeline import Pipeline
from collections import OrderedDict
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

patterns = {
    'english': r'\b[^\d\W]+\b',
    'alphanumeric': r'(?!\b\d+\b)\b\w+\b',
    'alphanum_nohex': r'(?!\b\d+\b|0x[0-9a-f]+)\b\w+\b'
}

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't", 'behavior', 'error', 'file', 'il2', 'il4', 'interface', 'issue', 'juniper', 'junos',
              'log', 'name', 'net', 'pr', 'root', 'show', 'system', 'unexpected', 'version']


class PRTransformer:
    def __init__(self):
        self.synop_vectorizer = TfidfVectorizer(analyzer='word', decode_error='ignore',
                                                stop_words=stop_words)

        self.desc_vectorizer = Pipeline([
            ('vector', CountVectorizer(token_pattern=patterns['alphanum_nohex'], stop_words=stop_words)),
            ('normalization', TfidfTransformer())
        ])

        self.transformers = [('synop', self.synop_vectorizer, 0), ('desc', self.desc_vectorizer, 1),
                             ('component', DictVectorizer(), 2)]

        self.vector = ColumnTransformer(self.transformers, n_jobs=-1,
                                        transformer_weights={'synop': 0.5, 'desc': 0.25, 'component': 0.25})

    def fit(self, corpus):
        self.vector.fit(corpus)
        return self

    def transform(self, corpus):
        return self.vector.transform(corpus)

    def fit_transform(self, corpus):
        self.X_train = self.vector.fit_transform(corpus)
        return self.X_train


class NNModel:
    def __init__(self, model_name='nn_model'):
        self.model_name = model_name
        self.vector = PRTransformer()
        self.core = NearestNeighbors(n_jobs=-1, p=2, algorithm='auto')
        # Existing PR Numbers.
        self.train_y = []

    def fit(self, train_X, train_y):
        print('Transforming Text..')
        self.train_y = train_y
        X_train = self.vector.fit_transform(train_X)
        print('Fitting Model..')
        self.core.fit(X_train)
        return self

    def transform(self, test_X):
        print('Transforming..')
        return self.vector.transform(test_X)

    # Predict from NN core
    def predict(self, item, n_neighbors=5):
        similar_prs = []
        similarities = []
        gaps, neighbors = self.core.kneighbors(item, n_neighbors)
        for i, indices in enumerate(neighbors):
            similar_prs = [self.train_y[index] for index in indices]
            for gap in gaps[i]:
                similarities.append(round((1 - gap) * 100, 2))

        return dict(zip(similar_prs, similarities))

    def save(self, path=None):
        print('Saving Model..')
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'trained_models')
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self, path + '/' + self.model_name + '.joblib')
        print('Model saved at: ', path + '/' + self.model_name + '.joblib')

    def load(self, path=None):
        if not path:
            path = os.path.join(os.path.dirname(__file__),
                                'trained_models/{model_name}.joblib'.format(model_name=self.model_name))
        print('Loading Model:', path)
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                self.vector = model.vector
                self.core = model.core
                self.train_y = model.train_y
                return self
            else:
                print('Model file does not exist on given path')
        except Exception as e:
            print(e)

        return self

    def get_X_y(self, pr_data):
        X = []
        y = []
        pbar = tqdm(total=len(pr_data))
        for pr, properties in pr_data.items():
            text = self.extract_columns(properties)
            X.append(text)
            y.append(pr)
            pbar.update(1)
        pbar.close()
        return X, y

    def extract_columns(self, properties):
        """
        Returns list made of synopsis (str), description (str), component (dict)
        :param properties:
        :return: list
        """
        description = properties.get('pr_description', '')
        if not description:
            description = ''
        description = description.split(' ')
        description = list(filter(None, description))
        if len(description) > 256:
            description = ' '.join(description[:256])
        else:
            description = ' '.join(description)

        return [properties.get('synopsis'), description, {'component': properties.get('category', '')}]

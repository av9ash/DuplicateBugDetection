from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import joblib
from tqdm import tqdm


class BERTEncoder(TransformerMixin, BaseEstimator):
    """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

    def __init__(self, path='all-mpnet-base-v2'):
        self.bert_model = SentenceTransformer(path)

    def fit(self, X):
        return self

    def transform(self, X):
        vectors = []
        pbar = tqdm(total=len(X))
        for x in X:
            vectors.append(self.bert_model.encode(x, convert_to_tensor=False).tolist())
            pbar.update(1)
        pbar.close()
        return np.array(vectors)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class BERT_NNModel:
    def __init__(self, model_name='bert_nn_model'):
        self.model_name = model_name
        self.model_dir = os.path.dirname(__file__)
        self.vector = BERTEncoder()
        self.core = NearestNeighbors(n_jobs=-1, p=2, algorithm='auto')
        # Existing PR Numbers.
        self.train_y = []

    def fit(self, train_X, train_y):
        print('\nTransforming Text..')
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

        text = properties.get('synopsis') + ' ' + description + ' ' + properties.get('category', '')
        return text

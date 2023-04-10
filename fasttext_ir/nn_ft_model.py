import fasttext
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import fasttext.util
from gensim.utils import simple_preprocess
from preprocessor import bugs_preprocessor
import os
import joblib


class FTNNModel:
    def __init__(self, model_name='ftnn_model'):
        self.ft_model = fasttext.load_model('fasttext_ir/crawl-300d-2M-subword.bin')
        self.model_name = model_name
        self.model_dir = os.path.dirname(__file__)
        self.core = NearestNeighbors(n_jobs=-1, p=2, algorithm='auto')
        # Existing PR Numbers.
        self.train_y = []

    def fit(self, train_X, train_y):
        print('Fitting Model..')
        self.train_y = train_y
        X_train = []
        for x in train_X:
            X_train.append(self.ft_model.get_sentence_vector(x))
        self.core.fit(X_train)
        return self

    def transform(self, test_X):
        print('Transforming..')
        X_test = []
        for x in test_X:
            X_test.append(self.ft_model.get_sentence_vector(x))
        return X_test

    # Predict from NN core
    def predict(self, item, n_neighbors=5):
        similar_prs = []
        similarities = []
        gaps, neighbors = self.core.kneighbors([item], n_neighbors)
        for i, indices in enumerate(neighbors):
            similar_prs = [self.train_y[index] for index in indices]
            for gap in gaps[i]:
                similarities.append(round((1 - gap) * 100, 2))

        return dict(zip(similar_prs, similarities))

    def get_X_y(self, pr_data):
        X = []
        y = []
        pbar = tqdm(total=len(pr_data))
        for pr, properties in pr_data.items():
            text_tokens = self.extract_columns(properties)
            X.append(text_tokens)
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
        tmp = bugs_preprocessor(text)
        text = ' '.join(tmp)
        return text

    def save(self, path=None):
        print('Saving Model..')
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'trained_models')
        if not os.path.exists(path):
            os.makedirs(path)

        self.ft_model = None
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
                self.core = model.core
                self.train_y = model.train_y
                if not self.ft_model:
                    self.ft_model = fasttext.load_model('fasttext_ir/crawl-300d-2M-subword.bin')
                return self
            else:
                print('Model file does not exist on given path')
        except Exception as e:
            print(e)

        return self

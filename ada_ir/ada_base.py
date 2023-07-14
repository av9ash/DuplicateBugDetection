
import json
import os
from sklearn.neighbors import NearestNeighbors
import joblib
from tqdm import tqdm
from preprocessor import bugs_preprocessor


class Ada_NNModel:
    def __init__(self, model_name='bert_nn_model'):
        self.model_name = model_name
        self.model_dir = os.path.dirname(__file__)
        self.core = NearestNeighbors(n_jobs=-1, p=2, algorithm='auto')
        # Existing PR Numbers.
        self.train_y = []
        self.embeddings = {}

    def fit(self, train_X, train_y):
        print('\nTransforming Text..')
        self.train_y = train_y

        for y in train_y:
            with open('../ada_ir/ada_embeddings/MozillaCore_training/{}_ada_embedding.json'.format(y)) as f:
                dkt = json.load(f)
                self.embeddings[dkt['pr_num']] = dkt['embedding']

        X_train = list(self.embeddings.values())

        print('Fitting Model..')
        self.core.fit(X_train)
        return self

    def transform(self, test_y):
        test_embeddings = []
        test_embedding_prs = [x.replace('_ada_embedding.json', '') for x in
                              os.listdir('../ada_ir/ada_embeddings/MozillaCore_testing')]
        for pr in test_y:
            with open('../ada_ir/ada_embeddings/MozillaCore_testing/{}_ada_embedding.json'.format(pr)) as f:
                dkt = json.load(f)
                test_embeddings.append(dkt['embedding'])

        return test_embeddings

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

    def save(self, path=None):
        print('Saving Model..')
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'trained_models')
        if not os.path.exists(path):
            os.makedirs(path)

        # Avoid saving embeddings
        self.embeddings = []
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
        tmp = bugs_preprocessor(text)
        text = ' '.join(tmp)

        return text

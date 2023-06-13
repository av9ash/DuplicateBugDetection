import os
import numpy as np
import joblib
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from gensim.utils import simple_preprocess


class GenModel:
    def __init__(self, model_name='d2v_model'):
        self.model_name = model_name
        self.model_gj = None
        self.model_dir = os.path.dirname(__file__)
        # Existing PR Numbers.
        self.train_y = []
        self.embeddings = {}

    def fit(self, train_X, train_y):
        print('Transforming Text..')
        self.train_y = train_y

        documents = []
        for idx, pr_num in enumerate(train_y):
            documents.append(TaggedDocument(train_X[idx], [pr_num]))

        cpu_count = multiprocessing.cpu_count() - 2
        self.model_gj = Doc2Vec(vector_size=300, window=5, min_count=2, workers=cpu_count)
        self.model_gj.build_vocab(corpus_iterable=documents)
        word_vectors = self.model_gj.wv
        word_vectors.vectors_lockf = np.ones(len(word_vectors))
        print('Fitting Model..')
        word_vectors.intersect_word2vec_format('gensim_ir/GoogleNews-vectors-negative300.bin', lockf=1.0,
                                               binary=True)
        print('Training..')
        self.model_gj.train(documents, total_examples=self.model_gj.corpus_count, epochs=100)
        self.embeddings = dict(zip(self.model_gj.dv.index_to_key, self.model_gj.dv.vectors))

        return self

    def transform(self, test_X):
        print('Transforming..')
        X_test = []
        for test_doc in test_X:
            X_test.append(self.model_gj.infer_vector(test_doc))
        return X_test

    # Predict from NN core
    def predict(self, item, n_neighbors=5):
        result = self.model_gj.dv.most_similar([item], topn=n_neighbors)
        result = [(pr, round(sim, 2)) for pr, sim in result]
        similar_prs = dict(result)
        return similar_prs

    def save(self, path=None):
        print('Saving Model..')
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'trained_models')
        if not os.path.exists(path):
            os.makedirs(path)

        save_loc = path + '/' + self.model_name + '.model'
        self.model_gj.save(save_loc)
        print('Model saved at: ', save_loc)

    def load(self, path=None):
        if not path:
            path = os.path.join(os.path.dirname(__file__),
                                'trained_models/{model_name}.model'.format(model_name=self.model_name))
        print('Loading Model:', path)
        try:
            if os.path.exists(path):
                self.model_gj = Doc2Vec.load(path)
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
        return simple_preprocess(text)

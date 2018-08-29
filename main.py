import logging
import random
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

from flask import Flask
from flask import jsonify


MODEL_NAME = 'iris_LR'
TRAIN_PCT = 0.7

app = Flask(__name__)

@app.route('/')
def home():
    """Welcome Screen"""
    return 'Welcome'


@app.route('/ok')
def ok():
    """Health Check"""
    return 'ok'


@app.route('/train_model')
def train_model():
    """Train simple model, save as pickle file."""

    # load iris through the scikit module
    logging.info('[main.train_model] getting iris dataframe')
    df = get_iris()

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # perform train/test split..
    split_pt = int(TRAIN_PCT * len(df))

    train_x = df[:split_pt].drop(u'target', 1)  # training features
    train_y = df[:split_pt].target  # training target

    test_x = df[split_pt:].drop(u'target', 1)  # test features
    test_y = df[split_pt:].target  # test target

    logging.info('[main.train_model] fitting model for %s rows', len(train_x))
    # fit the model to the dataset
    model = LR()
    model.fit(train_x, train_y)

    predicted_y = model.predict(test_x)
    cm = confusion_matrix(test_y, predicted_y)

    model_outputs = {
        'confusion_matrix': cm.__str__(),
        'accuracy': model.score(test_x, test_y),
        'coeffs': model.coef_[0].__str__(),
    }

    logging.info('[main.train_model] model_outputs - %s', model_outputs)

    filename = '/tmp/{model_name}.pkl'.format(model_name='iris_LR')
    pickle.dump(model, open(filename, 'wb'))

    return jsonify(model_outputs)


@app.route('/predict_random', methods=['GET'])
def predict_random():
    """
    Generate a random series for each of our features and pass that to the
    predict function,  This is used for load testing, to mock out what a live
    API would generate and send to this service
    """

    # iris = load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data = [[get_ran_float() for x in range(4)]]

    logging.info('[predict_random] %s ', data)

    try:
        prediction = predict(data, features)
    except Exception as err:
        logging.error('[predict_random] err - %s ', err)
        return jsonify({'err': 'err'})

    return jsonify({'predicted_class': prediction[0]})


def predict(data, features):
    """load the model from disk, make the prediction"""
    filename = '/tmp/{model_name}.pkl'.format(model_name=MODEL_NAME)
    loaded_model = pickle.load(open(filename, 'rb'))

    test_df = pd.DataFrame(data, columns=features)

    return loaded_model.predict(test_df)

def get_iris():

    iris = load_iris()
    rows = np.c_[iris['data'], iris['target']]
    columns = iris['feature_names'] + ['target']
    df = pd.DataFrame(data=rows, columns=columns).dropna()

    df[['target']] = df[['target']].astype(str)

    # map the enum to the species class
    lookup = {'0.0': 'Setosa', '1.0': 'Versicolour', '2.0': 'Virginica'}
    df['target'] = df['target'].map(lambda k: lookup.get(k, ''))

    return df

def get_ran_float():

    return random.uniform(1.0, 4.0)


if __name__ == '__main__':
    # for local development
    app.run(host='127.0.0.1', port=8080, debug=True)

# https://cloud.google.com/appengine/docs/standard/python3/quickstart
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://cloud.google.com/appengine/docs/standard/python3/runtime

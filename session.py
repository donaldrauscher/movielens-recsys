import zipfile

import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

from tensorrec import TensorRec
from tensorrec.eval import precision_at_k, recall_at_k
from tensorrec.representation_graphs import NormalizedLinearRepresentationGraph
from tensorrec.loss_graphs import WMRBLossGraph


import logging
logging.getLogger().setLevel(logging.INFO)


def load_data(path):
    """
    Return movielens 100K dataset (https://grouplens.org/datasets/movielens/100k/)
    """

    with zipfile.ZipFile(path) as datafile:
        return (
            pd.read_csv(datafile.open("ml-100k/ua.base"), delimiter='\t', header=None, names=['uid', 'iid', 'rating', 'timestamp']),
            pd.read_csv(datafile.open("ml-100k/ua.test"), delimiter='\t', header=None, names=['uid', 'iid', 'rating', 'timestamp']),
            pd.read_csv(datafile.open("ml-100k/u.item"), delimiter='|', header=None, encoding="ISO-8859-1",
                        names=['iid', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action',\
                               'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',\
                               'film_noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western'],
                        parse_dates=['release_date', 'video_release_date'])
        )


def create_sessions(df, n_fold=1, train_size=0.5):
    """
    Split train data into two pieces (n_fold times) for features and interactions
    """
    features = []
    interactions = []

    for i in range(n_fold):
        tmp = df.copy()
        tmp.uid = tmp.uid + df.uid.max()*i
        tmp1, tmp2 = train_test_split(tmp, train_size=train_size, stratify=tmp.uid, random_state=i)
        features.append(tmp1)
        interactions.append(tmp2)

    return (
        pd.concat(features, axis=0, ignore_index=True, sort=False),
        pd.concat(interactions, axis=0, ignore_index=True, sort=False)
    )


if __name__ == '__main__':

    # load raw data
    train, test, _ = load_data('ml-100k.zip')

    # number of users and items
    n_users = train.uid.max()
    n_items = train.iid.max()

    # split train into user features and interactions
    n_fold = 10
    train1, train2 = create_sessions(train, n_fold=10)

    format_rating = lambda x: 1 if x >= 4 else -1

    user_features = sp.coo_matrix((train1.rating.apply(format_rating), (train1.uid - 1, train1.iid - 1)),
                                  shape=(n_users*n_fold, n_items))

    train_interactions = sp.coo_matrix((train2.rating.apply(format_rating), (train2.uid-1, train2.iid-1)),
                                       shape=(n_users*n_fold, n_items))

    # create item features
    item_features = sp.identity(n_items)

    # create test stuff
    test_user_features = sp.coo_matrix((train.rating.apply(format_rating), (train.uid - 1, train.iid - 1)),
                                       shape=(n_users, n_items))

    test_interactions = sp.coo_matrix((test.rating.apply(format_rating), (test.uid-1, test.iid-1)),
                                      shape=(n_users, n_items))

    # train collaborative filtering model
    epochs = 500
    alpha = 0.00001
    n_components = 10
    verbose = True
    learning_rate = 0.01
    n_sampled_items = int(n_items*0.01)
    fit_kwargs = {'epochs': epochs, 'alpha': alpha, 'verbose': verbose, 'learning_rate': learning_rate,
                  'n_sampled_items': n_sampled_items}

    cf_model = TensorRec(n_components=10,
                         user_repr_graph=NormalizedLinearRepresentationGraph(),
                         loss_graph=WMRBLossGraph())

    cf_model.fit(user_features=user_features, item_features=item_features,
                 interactions=train_interactions, **fit_kwargs)

    # calculate test ranks excluding training items
    predicted_ranks = cf_model.predict_rank(user_features=test_user_features, item_features=item_features)
    predicted_ranks[train.uid - 1, train.iid - 1] = n_items + 1
    predicted_ranks = predicted_ranks.argsort(axis=1).argsort(axis=1) + 1

    # evaluate precision and recall
    precision_results = precision_at_k(predicted_ranks, test_interactions, k=10)
    recall_results = recall_at_k(predicted_ranks, test_interactions, k=10)

    logging.info("Precision at 10: {}".format(np.mean(precision_results)))
    logging.info("Recall at 10: {}".format(np.mean(recall_results)))

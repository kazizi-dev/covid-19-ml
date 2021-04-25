from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import recall_score, f1_score

import os, warnings, pickle
import pandas as pd
import numpy as np
import datetime

from helper import clean_age_data, impute_age_data, clean_sex_data
from helper import clean_date, clean_cols, remove_unused_cols, handle_skewed_data



def split_dataset(df):
    # label encode date_confirmation attribute:
    df['date_confirmation'] = LabelEncoder().fit_transform(df['date_confirmation'].astype(str))

    y_train = df[['outcome']].values
    y_train = y_train.ravel()

    # one hot encode to prevent correlation
    enc = OneHotEncoder()
    x_train = df.drop(['outcome'], axis=1).copy()
    categorical_data = x_train[['sex', 'country', 'province']]   
    print('1111111111111')
    binary_data = enc.fit_transform(categorical_data).toarray()
    binary_labels = np.append(enc.categories_[0], enc.categories_[1])
    binary_labels = np.append(binary_labels, enc.categories_[2])

    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x_train = x_train.drop(['sex', 'country', 'province'], axis=1)

    # append converted data and numerical data
    x_train = x_train.join(encoded_df)
    print('2222222222222')

    from pprint import pprint
    pprint(y_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, 
                                                        y_train, 
                                                        test_size=0.2, 
                                                        random_state=0,
                                                        stratify=y_train)
    return x_train, y_train, x_test, y_test


def print_classification_report(model, x_train, y_train, x_test, y_test):
    print('--- Classification report for train data:')
    y_pred = model.predict(x_train)
    print(classification_report(y_train, y_pred))

    print('--- Classification report for test data:')
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))


def get_grid_search_cv(x_train, y_train, model):
    params = {
        'n_estimators': [10], 
        'learning_rate': [1, 0.5, 0.1], 
        'algorithm': ['SAMME', 'SAMME.R']
    }

    scoring = {
        'f1_score_on_deceased' : make_scorer(f1_score, average='micro', labels=['deceased']),
        'recall_on_deceased' : make_scorer(recall_score, average='micro', labels=['deceased']),
        'overall_accuracy': make_scorer(accuracy_score),
        'overall_recall': make_scorer(recall_score , average='weighted')
    }

    gs = GridSearchCV(
        model, 
        param_grid = params, 
        scoring = scoring, 
        n_jobs = -1, 
        cv = 3,
        refit = 'f1_score_on_deceased'
    )

    gs.fit(x_train, y_train)
    return gs


def get_ada_boost_results(csv_path):
    warnings.filterwarnings("ignore")
    os.chdir(os.getcwd())

    ########### split and encode data ###########
    df = pd.read_csv(csv_path)

    print("...splitting and encoding data")
    x_train, y_train, x_test, y_test = split_dataset(df)

    ########### get best parameters ###########
    print("...tunning ada boost model using GridSearchCV")

    ada_model = AdaBoostClassifier()
    gs = get_grid_search_cv(x_train, y_train, ada_model)

    gs_results = pd.DataFrame(gs.cv_results_)[['mean_fit_time', 
                                                'param_algorithm',
                                                'param_learning_rate',
                                                'param_n_estimators', 
                                                'mean_test_f1_score_on_deceased', 
                                                'rank_test_f1_score_on_deceased', 
                                                'mean_test_overall_accuracy', 
                                                'rank_test_overall_accuracy', 
                                                'mean_test_recall_on_deceased', 
                                                'rank_test_recall_on_deceased', 
                                                'mean_test_overall_recall', 
                                                'rank_test_overall_recall']]

    gs_results.to_csv('../results/tunning-results-ada-boost.csv')

    ########### train and predict with model ###########
    print('\n***************************** Ada Boost Results *****************************')
    print(f'Best Parameters: {gs.best_params_}')
    
    ada_model = gs.best_estimator_

    ########### save the model ###########
    print('...saving model as pickel file')
    path = '../models/ada_boost_model.pkl'
    pickle.dump(ada_model, open(path, 'wb'))
    ada_model = pickle.load(open(path, 'rb'))




### manually train the model on a given parameters NE and LR
def test(csv_path, NE, LR):
    import warnings
    warnings.filterwarnings("ignore")

    # setup path for main.py file
    os.chdir(os.getcwd())

    # read the processed data
    df = pd.read_csv(csv_path)

    # split the dataset to train and test data
    print("...splitting and encoding data")
    x_train, y_train, x_test, y_test = split_dataset(df)

    print('\n***************************** Ada Boost Results *****************************')
    ada_model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=NE, learning_rate=LR)
    ada_model.fit(x_train, y_train)

    print_classification_report(ada_model, x_train, y_train, x_test, y_test)

    print(f'Train Accuracy: {ada_model.score(x_train, y_train)}')
    print(f'Test Accuracy: {ada_model.score(x_test, y_test)}')

    print('...saving model as pickel file')
    path = '../models/ada_boost_model_test.pkl'
    pickle.dump(ada_model, open(path, 'wb'))
    ada_model = pickle.load(open(path, 'rb'))

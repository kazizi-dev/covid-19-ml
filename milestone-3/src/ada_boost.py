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

from helper import impute_age_data, clean_sex_data
from helper import clean_date, clean_num_cols, remove_unused_cols


def split_dataset(df):
    """
    encode string attributes to convert to numeric values using label and one hot encoder
    """
    df = df.drop(df.columns[0], axis=1)
    # label encode date_confirmation attribute:
    df['date_confirmation'] = LabelEncoder().fit_transform(df['date_confirmation'].astype(str))

    y_train = df[['outcome']].values
    y_train = y_train.ravel()

    # one hot encode to prevent correlation
    enc = OneHotEncoder()
    x_train = df.drop(['outcome'], axis=1).copy()
    categorical_data = x_train[['sex']]   
    binary_data = enc.fit_transform(categorical_data).toarray()
    binary_labels = enc.categories_[0]

    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x_train = x_train.drop(['sex', 'country', 'province'], axis=1)

    # append converted data and numerical data
    x_train = x_train.join(encoded_df)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
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
        'n_estimators': [10, 100, 2500], 
        'learning_rate': [1, 0.5, 0.1], 
        'algorithm': ['SAMME', 'SAMME.R']
    }

    scoring = {
        'f1_score_on_deceased' : make_scorer(f1_score, average='weighted', labels=['deceased']),
        'recall_on_deceased' : make_scorer(recall_score, average='weighted', labels=['deceased']),
        'overall_accuracy': make_scorer(accuracy_score),
        'overall_recall': make_scorer(recall_score , average='weighted')
    }

    gs = GridSearchCV(
        model, 
        param_grid = params, 
        scoring = scoring, 
        n_jobs = -1, 
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

    print("************************************ end ************************************\n")


def test_cases_test(csv_path):
    import warnings
    warnings.filterwarnings("ignore")

    # setup path for main.py file
    os.chdir(os.getcwd())

    # Get cases dataset
    filename = '../data/cases_test.csv'
    cases_csv = open(filename, 'rt')
    cases_df = pd.read_csv(filename)
    cases_csv.close()
    
    # Get transformed location dataset
    filename = '../data/location_transformed.csv'
    loc_csv = open(filename, 'rt')
    loc_df = pd.read_csv(filename)
    loc_csv.close()
    
    # Join location and cases datasets
    cases_df = pd.merge(cases_df, loc_df, how="left")

    clean_sex_data(cases_df)
    impute_age_data(cases_df.age.tolist(), cases_df)
    cases_df = remove_unused_cols(cases_df)
    clean_date(cases_df)
    clean_num_cols(cases_df, ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio'])
    cases_df = cases_df.drop(['country', 'province'], axis=1)
    
    # writing to CSV
    result_filename = '../data/cases_test_processed.csv'
    cases_df.to_csv(result_filename, index=False)


def make_prediction(model, test_df, SYSTEM_INPUT):   
    
    predictions = model.predict(test_df)
    
    test_df['outcome'] = predictions
    
    # csv with labels
    test_df.to_csv('../results/cases_test_predictions.csv', index=False)
    
    # save to txt file
    np.savetxt('../results/adaboost-predictions.txt', predictions, fmt='%s')
    
    # fix newline 
    # from SO: https://stackoverflow.com/questions/28492954/numpy-savetxt-stop-newline-on-final-line
    with open('../results/adaboost-predictions.txt', 'w') as fout:
        NEWLINE_SIZE_IN_BYTES = SYSTEM_INPUT  
        np.savetxt(fout, predictions, fmt='%s') # Use np.savetxt.
        fout.seek(0, os.SEEK_END) # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        fout.truncate() # Truncate the file to this point.


def encode_predict_adaboost(SYSTEM_INPUT):
    # Process cases_test
    data_path = '../data/cases_test_processed.csv' # name of processed test data
    test_cases_test(data_path)
    
    # Load Model
    path = '../models/ada_boost_model.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    # Load Data
    df = pd.read_csv('../data/cases_test_processed.csv')
    
    # Encoding joined cases_test data
    df['date_confirmation'] = LabelEncoder().fit_transform(df['date_confirmation'].astype(str))
    
    enc = OneHotEncoder()
    x = df.drop(['outcome'], axis=1).copy()
    categorical_data = x[['sex']]   
    binary_data = enc.fit_transform(categorical_data).toarray()
    binary_labels = enc.categories_[0]
    
    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x = x.drop(['sex'], axis=1)
    x = x.join(encoded_df)
    
    x.to_csv('../data/cases_test_encoded.csv', index=False)
    
    make_prediction(model, x, SYSTEM_INPUT)


def check_if_file_valid(filename):
    assert filename.endswith('adaboost-predictions.txt'), 'Incorrect filename'
    f = open(filename).read()
    l = f.split('\n')
    assert len(l) == 46500, 'Incorrect number of items'
    assert (len(set(l)) == 4), 'Wrong class labels'
    return 'The predictions file is valid'


# ============================================================================================

def start_testing_ada_boost(SYSTEM_INPUT):
    # Train a model on different hyperparams
    get_ada_boost_results('../data/cases_train_processed.csv')

    # Predict on test data
    encode_predict_adaboost(SYSTEM_INPUT)

    check_if_file_valid('../results/adaboost-predictions.txt')
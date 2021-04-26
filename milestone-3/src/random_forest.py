import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from sklearn import metrics as met
from matplotlib import pyplot

from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, precision_score

import warnings

from helper import clean_age_data, impute_age_data, clean_sex_data
from helper import clean_date, clean_cols, clean_num_cols, remove_unused_cols, handle_skewed_data

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
    #binary_labels = np.append(enc.categories_[0], enc.categories_[1])
    #binary_labels = np.append(binary_labels, enc.categories_[2])

    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x_train = x_train.drop(['sex', 'country', 'province'], axis=1)

    # append converted data and numerical data
    x_train = x_train.join(encoded_df)
    print(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test

def get_grid_search_cv(x_train, y_train, model):
    params= {  
        'n_estimators': [10, 100, 200, 400, 600, 800, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [25],
        'criterion' :['entropy']
    }

    scoring = {
        'f1_score_on_deceased' : make_scorer(f1_score, average='micro', labels=['deceased']),
        'overall_accuracy': make_scorer(accuracy_score),
        'recall_on_deceased' : make_scorer(recall_score, average='micro', labels=['deceased']),
        'overall_recall': make_scorer(recall_score , average='weighted')
    }

    gs = GridSearchCV(
        model, 
        param_grid = params, 
        scoring = scoring, 
        n_jobs = 2, 
        refit = 'f1_score_on_deceased'
    )

    gs.fit(x_train, y_train)
    return gs

def start_random_forest():
    warnings.filterwarnings("ignore")
    os.chdir(os.getcwd())

    ########### split and encode data ###########
    #df = pd.read_csv('../data/cases_.csv')
    df = pd.read_csv('../data/cases_train_processed.csv')
    #df = pd.read_csv('../data/cases_test_joined.csv')
    #df = df[:500]

    print("...splitting and encoding data")
    x_train, y_train, x_test, y_test = split_dataset(df)

    ########### get best parameters ###########
    print("...tunning random forest model using GridSearchCV")

    rf_model = RandomForestClassifier(random_state=0)
    gs = get_grid_search_cv(x_train, y_train, rf_model)

    gs_results = pd.DataFrame(gs.cv_results_)[['mean_fit_time',
                                                'param_n_estimators',
                                                'param_max_features',
                                                'param_max_depth', 
                                                'param_criterion',
                                                'mean_test_f1_score_on_deceased', 
                                                'rank_test_f1_score_on_deceased', 
                                                'mean_test_overall_accuracy', 
                                                'rank_test_overall_accuracy', 
                                                'mean_test_recall_on_deceased', 
                                                'rank_test_recall_on_deceased', 
                                                'mean_test_overall_recall', 
                                                'rank_test_overall_recall']]

    gs_results.to_csv('../results/tunning-results-random_forest.csv')

    ########### train and predict with model ###########
    print('\n***************************** Rnadom Forest Results *****************************')
    print(f'Best Parameters: {gs.best_params_}')
    
    rf_model = gs.best_estimator_

    ########### save the model ###########
    print('...saving model as pickel file')
    path = '../models/random_forest_model.pkl'
    pickle.dump(rf_model, open(path, 'wb'))
    rf_model = pickle.load(open(path, 'rb'))

    print("***************************** end *****************************\n")

def test_cases_test(csv_path):
    import warnings
    warnings.filterwarnings("ignore")

    # setup path for main.py file
    os.chdir(os.getcwd())

    #file_path = '../data/cases_test.csv'
    #cases_df = pd.read_csv(file_path)
    
    # ========================================
    # 1.5 JOINING DATASETS
    # ========================================
    
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
    #joint_df = pd.merge(cases_df, loc_df, how="left")
    #joint_df = joint_df.fillna(0)
    cases_df = pd.merge(cases_df, loc_df, how="left")

    ## display missing values for cases_train file
    print(f'---> null values for {filename} file:')
    print(cases_df.isnull().sum())

    clean_sex_data(cases_df)
    impute_age_data(cases_df.age.tolist(), cases_df)
    # cases_df = handle_skewed_data(cases_df)
    cases_df = remove_unused_cols(cases_df)
    clean_date(cases_df)
    clean_num_cols(cases_df, ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio'])
    #clean_cols(cases_df, ["country", "province"])
    cases_df = cases_df.drop(['country', 'province'], axis=1)
    
    # writing to CSV
    result_filename = '../data/cases_test_processed.csv'
    #joint_df.to_csv(result_filename, index=False)
    cases_df.to_csv(result_filename, index=False)

    

def make_prediction(model, test_df):   
    
    predictions = model.predict(test_df)
    
    test_df['outcome'] = predictions
    
    test_df.to_csv('../results/cases_test_predictions.csv', index=False)
    print()
    
def encode_predict():
    # process cases_test
    data_path = '../data/cases_test_processed.csv' # name of processed test data
    test_cases_test(data_path)
    
    # Load Model and Data
    
    path = '../models/random_forest_model.pkl'
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    df = pd.read_csv('../data/cases_test_processed.csv')
    
    
    # encoding joined cases_test data
    
    df['date_confirmation'] = LabelEncoder().fit_transform(df['date_confirmation'].astype(str))
    #df['additional_information'] = LabelEncoder().fit_transform(df['additional_information'].astype(str))
    
    y_labels = df[['outcome']]
    #y_labels = y_labels.ravel()
    
    enc = OneHotEncoder()
    x = df.drop(['outcome'], axis=1).copy()
    #categorical_data = x[['sex', 'country', 'province']]
    categorical_data = x[['sex']]   
    binary_data = enc.fit_transform(categorical_data).toarray()
    binary_labels = enc.categories_[0]
    #binary_labels = np.append(enc.categories_[0], enc.categories_[1])
    #binary_labels = np.append(binary_labels, enc.categories_[2])
    
    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x = x.drop(['sex'], axis=1)
    x = x.join(encoded_df)
    #x = x.join(y_labels)
    #x = x.fillna(x.mean())
    
    x.to_csv('../data/cases_test_encoded.csv', index=False)
    
    make_prediction(model, x)
    


# ============================================================================================


start_random_forest()
encode_predict()

print("~~~~~~~~~~end~~~~~~~~~~")

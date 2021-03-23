import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import csv


def split_dataset(df):
    """
    encode string attributes to convert to numeric values using label and one hot encoder
    """
    
    # label encode date_confirmation attribute:
    df['date_confirmation'] = LabelEncoder().fit_transform(df['date_confirmation'].astype(str))

    y_train = df[['outcome']].values
    y_train = y_train.ravel()

    # one hot encode to prevent correlation
    enc = OneHotEncoder()
    x_train = df.drop(['outcome', 'longitude', 'latitude', 'province'], axis=1).copy()
    categorical_data = x_train[['sex', 'country']]
    binary_data = enc.fit_transform(categorical_data).toarray()
    binary_labels = np.append(enc.categories_[0], enc.categories_[1])

    encoded_df = pd.DataFrame(binary_data, columns=binary_labels)
    x_train = x_train.drop(['sex', 'country'], axis=1)

    # append converted data and numerical data
    x_train = x_train.join(encoded_df)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test


def build_ada_boost_model(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8, random_state=0)
    model = model.fit(x_train, y_train)
    
    path = './models/ada_boost_model.pkl'
    pickle.dump(model, open(path, 'wb'))
    model = pickle.load(open(path, 'rb'))
    
    print('>> Ada Boost Model:')
    print(f'Train Accuracy: {model.score(x_train, y_train)}')
    print(f'Test Accuracy: {model.score(x_test, y_test)}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ 2.2 RANDOM FORESTS ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(x_train, y_train)

    path = './models/random_forst_model.pkl'
    pickle.dump(model, open(path, 'wb'))
    model = pickle.load(open(path, 'rb'))

    print('>> Random Forst Model:')
    print(f'Train Accuracy: {model.score(x_train, y_train)}')
    print(f'Test Accuracy: {model.score(x_test, y_test)}')


if __name__ == '__main__':
    print('--------------------------')
    # setup path for main.py file
    os.chdir(os.getcwd())

    # read the processed data
    df = pd.read_csv('./data/cases_train_processed.csv')

    # split the dataset to train and test data
    x_train, y_train, x_test, y_test = split_dataset(df)


    build_ada_boost_model(x_train, y_train, x_test, y_test)
    build_random_forest(x_train, y_train, x_test, y_test)
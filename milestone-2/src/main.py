import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import os
import pickle


def encode_dataset(df):
    """
    convert string attributes to numeric using label encoder
    """
    attributes = ['sex', 'province', 'country', 'date_confirmation', 'outcome']
    label_encode = LabelEncoder()

    for attribute in attributes:
        df[attribute] = label_encode.fit_transform(df[attribute].astype(str))


def split_dataset(df):
    """
    Split the dataset to 80:20
    """
    encode_dataset(df)

    x = df.drop('outcome', axis=1)
    y = df['outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, y_train, x_test, y_test


def build_ada_boost_model(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8, random_state=0)
    model = model.fit(x_train, y_train)
    
    path = './models/ada_boost_model.pkl'
    pickle.dump(model, open(path, 'wb'))
    model = pickle.load(open(path, 'rb'))
    
    print(f'Train Accuracy: {model.score(x_train, y_train)}')
    print(f'Test Accuracy: {model.score(x_test, y_test)}')

if __name__ == '__main__':
    print('--------------------------')
    # setup path for main.py file
    os.chdir(os.getcwd())

    # read the processed data
    df = pd.read_csv('./data/cases_train_processed.csv')

    x_train, y_train, x_test, y_test = split_dataset(df)


    build_ada_boost_model(x_train, y_train, x_test, y_test)

    # df.to_csv('./data/test.csv', index=False)
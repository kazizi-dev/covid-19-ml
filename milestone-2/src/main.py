import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import os


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
    x_train, x_test, y_train, y_test = train_test_split(df['outcome'], 
                                                        df.drop('outcome', axis=1), 
                                                        test_size=0.2, 
                                                        random_state=0)

    # return two tuples
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    print('--------------------------')
    # setup path for main.py file
    os.chdir(os.getcwd())

    # read the processed data
    df = pd.read_csv('./data/cases_train_processed.csv')

    x_train, y_train, x_test, y_test = split_dataset(df)


    # df.to_csv('./data/test.csv', index=False)
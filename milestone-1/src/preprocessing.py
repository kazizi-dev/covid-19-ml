# impute missing age values by replacing 0 with mean value the dataset
import pandas as pd
import numpy as np
import math
import datetime
from sklearn.impute import SimpleImputer


def clean_age(df):
    """
    clean the entries for the age attribute
    """

    arr = df['age'].tolist()

    # clean entries and convert string values to integer
    for i in range(len(arr)):
        if arr[i] is np.nan:
            arr[i] = int(np.nan_to_num(arr[i]))
        if isinstance(arr[i], str):
            if '.' in arr[i]:               # str > float > int
                temp = int(round(float(arr[i])))
                arr[i] = temp if temp > 0 else -temp
            elif '-' in arr[i]:             # take avg of two numbers in string '15-34'
                split_arr = arr[i].split('-')
                t0 = split_arr[0]
                t1 = split_arr[1]
                if str.isnumeric(t0) == True and str.isnumeric(t1) == True:
                    avg = (round((int(t0) + int(t1)) / 2)) if len(t1) > 0 else int(t0)
                    arr[i] = avg if avg > 0 else -avg
                else:
                    arr[i] = 0
            elif '+' in arr[i]:             # convert strings with '+' to int
                split_arr = arr[i].split('+')
                num = int(split_arr[0])
                arr[i] = num if num > 0 else -num
            elif 'month' in arr[i]:         # convert months to years
                split_arr = arr[i].split(' ')
                arr[i] = round(int(split_arr[0]) / 12)
            else:                         # convert string int to actual int
                arr[i] = int(arr[i])
        elif isinstance(arr[i], float):   # round float values
                arr[i] = int(round(arr[i]))
        elif isinstance(arr[i], int):     # convert negative int to positive
            arr[i] = -arr[i] if arr[i] < 0 else arr[i]

    # impute missing entries for the age attribute using sklearn SimpleImputer
    df['age'] = arr

    val = 0
    if val in df['age'].values.tolist():
        imputer = SimpleImputer(missing_values=val, strategy='median')
        df.age = imputer.fit_transform(df['age'].values.reshape(-1,1))[:,0]


def clean_sex(df):
    """
    clean the entries for the sex attribute
    """
    
    df['sex'] = df['sex'].replace(np.nan, 'unknown', regex=True)


def clean_date(df):
    """
    clean the entries for the date attribute
    """

    arr = df['date_confirmation'].tolist()

    # convert to standard date format
    for i, entry in enumerate(arr):
        if isinstance(entry, str) and '.' in entry:
            split_arr = entry.replace(" ","").split('-')
            entry = split_arr[0] if len(split_arr) == 1 else split_arr[1]
            d = datetime.datetime.strptime(entry, '%d.%m.%Y')
            arr[i] = d.strftime('%Y-%m-%d')

    df['date_confirmation'] = arr

    # replace null entries with the mean value
    mean = (np.array(df['date_confirmation'].dropna(), dtype='datetime64[s]')
            .view('i8')
            .mean()
            .astype('datetime64[s]')).astype(str)

    df['date_confirmation'] = df['date_confirmation'].replace(np.nan, mean[0:10])
    

# handle null values for country and province attributes
def clean_cols(df, attributes=["country", "province"]):
    """
    replace the null entries of multiple columns
    """
    for attribute in attributes: 
        if df[attribute].isnull().sum() > 0:
            df[attribute] = df[attribute].replace(np.nan, 'none')

    # drop the null values of location coordinates
    df.dropna(subset=['latitude'], inplace=True)
    df.dropna(subset=['longitude'], inplace=True)


def drop_unused_columns(df):     
    """
    drop the data for these attributes because we don't use it
    """  

    if 'source' in df.columns:
        df.drop('source', axis = 1, inplace=True)
    if 'additional_information' in df.columns:
        df.drop('additional_information', axis = 1, inplace=True)
    else:
        print("[INFO]: attributes do not exist or might have been deleted.")


# TODO
def handle_skewed_data(df):
    pass

# TODO
def transform_and_join(df):
    pass
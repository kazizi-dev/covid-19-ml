import os, warnings, pickle
from pprint import pprint
import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer


def clean_age_data(arr):
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

    return arr


# impute missing age values by replacing 0 with mean value the dataset
def impute_age_data(arr, cases_df):
    cases_df['age'] = clean_age_data(arr)
    
    ## Use SimpleImputer from Sklearn library to impute missing age data
    ## to do this use the mean strategy by replacing NaN values by the mean
    ## of the dataset
    
    val = 0
    if val in cases_df['age'].values.tolist():
        imputer = SimpleImputer(missing_values=val, strategy='median')
        cases_df.age = imputer.fit_transform(cases_df['age'].values.reshape(-1,1))[:,0]
    else:
        print("Message: Impute already done")
        

# replace null entries by non-binary
def clean_sex_data(cases_df):
    cases_df['sex'] = cases_df['sex'].replace(np.nan, 'unknown', regex=True)


# clean null values for the attribute
def clean_date(cases_df):
    arr = cases_df['date_confirmation'].tolist()
    for i, entry in enumerate(arr):
        if isinstance(entry, str) and '.' in entry:
            split_arr = entry.replace(" ","").split('-')
            entry = split_arr[0] if len(split_arr) == 1 else split_arr[1]
            d = datetime.datetime.strptime(entry, '%d.%m.%Y')
            arr[i] = d.strftime('%Y-%m-%d')

    cases_df['date_confirmation'] = arr

    # use the mean date to replace with nan
    mean = (np.array(cases_df['date_confirmation'].dropna(), dtype='datetime64[s]')
        .view('i8')
        .mean()
        .astype('datetime64[s]')).astype(str)

    cases_df['date_confirmation'] = cases_df['date_confirmation'].replace(np.nan, 
                                                              mean[0:10])
    

# handle null values for country and province attributes
def clean_cols(cases_df, attributes=["country", "province"]):
    for attribute in attributes: 
        if cases_df[attribute].isnull().sum() > 0:
            cases_df[attribute] = cases_df[attribute].replace(np.nan, 'none')

    # handle null values for lattitude and longitude attributes
    cases_df.dropna(subset=['latitude'], inplace=True)
    cases_df.dropna(subset=['longitude'], inplace=True)

def clean_num_cols(cases_df, attributes=['Active', 'Deaths', 'Confirmed', 'Recovered', 'Incidence_Rate', 'Case-Fatality_Ratio']):
    for attribute in attributes: 
        if cases_df[attribute].isnull().sum() > 0:
            cases_df[attribute] = cases_df[attribute].replace(np.nan, 0)

# drop unused columns
def remove_unused_cols(df):       
    if 'source' in df.columns:
        df = df.drop('source', axis = 1)
    if 'additional_information' in df.columns:
        df = df.drop('additional_information', axis = 1)
    else:
        print("Message: columns do not exist!")

    return df


# dela with outliers
def handle_skewed_data(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # print(IQR)
    longa = df['longitude'].quantile(0.10)
    longb = df['longitude'].quantile(0.90)
    lata = df['latitude'].quantile(0.10)
    latb = df['latitude'].quantile(0.90)
    df["longitude"] = np.where(df["longitude"] <longa, longa,df['longitude'])
    df["longitude"] = np.where(df["longitude"] >longb, longb,df['longitude'])
    df["latitude"] = np.where(df["latitude"] <lata, lata,df['latitude'])
    df["latitude"] = np.where(df["latitude"] >lata, lata,df['latitude'])
    df['longitude'].skew()
    df['latitude'].skew()
    index = df[(df['age'] >= 115)|(df['age'] <= 0)].index
    df.drop(index, inplace=True)
    df['age'].describe()
    
    return df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

import pandas as pd
import numpy as np
import math
import helper1 as h
import datetime
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_dir = os.getcwd()
os.chdir(current_dir[0:len(current_dir)-3])

print("---------------------- program start ----------------------")

# ====================================
# 1.1 Data Analysis and visualizations
# ====================================

# histogram plot to demonstrade age distribution for male and female:
def plot_age_and_sex():
    color_idx = 0
    colors = ['orange', 'red', 'blue']
    for sex in cases_df['sex'].unique():
        plt.rcParams['figure.figsize'] = (10, 5)
        plt.title(f'Age Distribution for {sex}', size = 12)
        plt.xlabel('Age', size = 12)
        plt.ylabel('Count', size = 12)

        sns.histplot(cases_df['age'][cases_df['sex'] == sex],
                    color = colors[color_idx], 
                    label = 'Age', 
                    alpha=0.5,
                    linewidth=1)

        color_idx += 1
        plt.show()
        plt.savefig(f'./plots/age_distribution_for_{sex}.png')
        plt.close()
   

# show the percentage of sexes in a pie chart
def plot_sex_graph():
    counts = list(cases_df.sex.value_counts())
    labels = cases_df.sex.value_counts()
    labels = ['unknown', 'male', 'female']
    colors = ['lightgrey', 'lightblue', 'lightpink']
    plt.pie(counts, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=100)
    
    plt.title('Sex Percentage', size = 12)
    plt.show()
    plt.savefig('./plots/sex_distribution_pie_chart.png')
    plt.close()


# show the percentage of outcomes in a pie chart
def plot_outcome_graph():
    counts = list(cases_df.outcome.value_counts())
    labels = cases_df.outcome.value_counts()
    labels = cases_df.outcome.unique().tolist()
    colors = ['lightgreen', 'lightpink', 'lightblue', 'red']
    plt.pie(counts, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=100)
    
    plt.title('Outcome Percentage', size = 12)
    plt.show()
    plt.savefig('./plots/outcome_distribution_pie_chart.png')
    plt.close()


# display covid trend per outcome against time
def plot_cases_against_time():
    colors = ["green", "purple", "orange", "red"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    cases_df['date_confirmation'] = pd.to_datetime(cases_df['date_confirmation'])

    sns.lineplot(x='date_confirmation', 
                y=range(len(cases_df['outcome'])), 
                data=cases_df, 
                hue='outcome')
    
    plt.title('Covid case trend', size = 12)
    plt.ylabel('Total Cases')
    plt.xlabel('Date')
    plt.show()
    plt.savefig('./plots/covid_trend_per_outcome.png')
    plt.close()

# ====================================
# 1.2 Data Cleaning and Imputation
# ====================================

# convert str values to integer, and replace NaN values by 0
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
def impute_age_data(arr):
    cases_df['age'] = clean_age_data(arr)
    
    ## Use SimpleImputer from Sklearn library to impute missing age data
    ## to do this use the mean strategy by replacing NaN values by the mean
    ## of the dataset
    
    val = 0
    if val in cases_df['age'].values.tolist():
        imputer = SimpleImputer(missing_values=val, strategy='mean')
        cases_df.age = imputer.fit_transform(cases_df['age'].values.reshape(-1,1))[:,0]
    else:
        print("Message: Impute already done")
        

# replace null entries by non-binary
def clean_sex_data():
    cases_df['sex'] = cases_df['sex'].replace(np.nan, 'unknown', regex=True)


# clean null values for the attribute
def clean_date():
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
def clean_cols(attributes=["country", "province"]):
    for attribute in attributes: 
        if cases_df[attribute].isnull().sum() > 0:
            cases_df[attribute] = cases_df[attribute].replace(np.nan, 'none')

    # handle null values for lattitude and longitude attributes
    cases_df.dropna(subset=['latitude'], inplace=True)
    cases_df.dropna(subset=['longitude'], inplace=True)


# drop unused columns
def remove_unused_cols(df):       
    if 'source' in df.columns:
        df = df.drop('source', axis = 1)
    if 'additional_information' in df.columns:
        df = df.drop('additional_information', axis = 1)
    else:
        print("Message: columns do not exist!")

    return df


file_path = './data/cases_train.csv'
cases_df = pd.read_csv(file_path)

## display missing values for cases_train file
print(f'---> null values for {file_path} file:')
print(cases_df.isnull().sum())

clean_sex_data()
impute_age_data(cases_df.age.tolist())
cases_df = remove_unused_cols(cases_df)
clean_date()
clean_cols(["country", "province"])

result_filename = './results/cases_train_processed.csv'
cases_df.to_csv(result_filename, index=False)

plot_age_and_sex()
plot_sex_graph()
plot_outcome_graph()
plot_cases_against_time()


file_path = './data/cases_test.csv'
cases_df = pd.read_csv(file_path)

## display missing values for cases_train file
print(f'---> null values for {file_path} file:')
print(cases_df.isnull().sum())

clean_sex_data()
impute_age_data(cases_df.age.tolist())
cases_df = remove_unused_cols(cases_df)
clean_date()
clean_cols(["country", "province"])

result_filename = './results/cases_test_processed.csv'
cases_df.to_csv(result_filename, index=False)
# ====================================
# 1.4 TRANSFORMATION
# ====================================

# Get location dataset

filename = './data/location.csv'
loc_csv = open(filename, 'rt')
loc_df = pd.read_csv(filename)
loc_csv.close()

print(f'---> null values for {filename} file:')
print(loc_df.isnull().sum())


# Keep only desired columns
trloc_df = loc_df[["Province_State",
                   "Country_Region",
                   "Confirmed",
                   "Deaths",
                   "Recovered",
                   "Active",
                   "Incidence_Rate",
                   "Case-Fatality_Ratio"]]

# Go through each attribute in Country Region
# For each new region, = new row
# For each row, combine other attributes

rows = []

# Group by country
country_labels = trloc_df.Country_Region.unique()
country_groups = trloc_df.groupby("Country_Region")

for country in country_labels:
    
    # Check for separate provinces  
    country_df = country_groups.get_group(country)
    province_labels = country_df.Province_State.unique()

    # Checks last value for nan
    # This is because of how provinces are ordered in location.csv
    # Countries containing Nan provinces
    # are treated as having no provinces
    if (pd.notna(province_labels[-1])):

        # Has provinces  
        for province in province_labels:

            province_groups = country_df.groupby("Province_State")
            province_df = province_groups.get_group(province)         

            new_row = h.combine_rows(province_df)
            rows.append(new_row)
        
    else:
        
        # Does not have provinces
        new_row = h.combine_rows(country_df)
        rows.append(new_row)

new_df = pd.DataFrame(rows)

# Rename columns to prepare for joining
new_df.rename(columns={"Province_State": "province",
                       "Country_Region": "country"}, inplace=True)

# writing to CSV
result_filename = './results/location_transformed.csv'
new_df.to_csv(result_filename, index=False)

# ========================================
# 1.5 JOINING DATASETS
# ========================================

# Get cases dataset
filename = './results/cases_train_processed.csv'
cases_csv = open(filename, 'rt')
cases_df = pd.read_csv(filename)
cases_csv.close()

# Get transformed location dataset
filename = './results/location_transformed.csv'
loc_csv = open(filename, 'rt')
loc_df = pd.read_csv(filename)
loc_csv.close()

# Join location and cases datasets
joint_df = pd.merge(cases_df, loc_df, how="left")

# writing to CSV
result_filename = './results/cases_joined.csv'
joint_df.to_csv(result_filename, index=False)

print("---------------------- program ended ----------------------")

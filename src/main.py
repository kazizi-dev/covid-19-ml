import pandas as pd
import numpy as np
import math
import helper1 as h
import os

current_dir = os.getcwd()
os.chdir(current_dir[0:len(current_dir)-3])

print("program start")

# ====================================
# 1.4 TRANSFORMATION
# ====================================

# Get location dataset

filename = './data/location.csv'
loc_csv = open(filename, 'rt')
loc_df = pd.read_csv(filename)
loc_csv.close()

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
print("before rename:")
print(new_df.iloc[0])
new_df.rename(columns={"Province_State": "province",
                       "Country_Region": "country"}, inplace=True)
print("after rename:")
print(new_df.iloc[0])

# writing to CSV
result_filename = './results/location_transformed.csv'
new_df.to_csv(result_filename, index=False)

# ========================================
# 1.5 JOINING DATASETS
# ========================================

# Get cases dataset
filename = './data/cases_train.csv'
cases_csv = open(filename, 'rt')
cases_df = pd.read_csv(filename)
cases_csv.close()

# Get transformed location dataset
filename = './data/location_transformed.csv'
loc_csv = open(filename, 'rt')
loc_df = pd.read_csv(filename)
loc_csv.close()

# Join location and cases datasets
joint_df = pd.merge(cases_df, loc_df, how="left")

# writing to CSV
result_filename = './results/cases_joined.csv'
joint_df.to_csv(result_filename, index=False)

print("program end")

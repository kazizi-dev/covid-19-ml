import pandas as pd
#import numpy as np
#import math
import milestone1_helper as h
print("program start")

# ====================================
# 1.4 TRANSFORMATION
# ====================================

# ------------------
# 1.4 functions
# ------------------
'''
def combine_rows(grouped_df):

    # Condense df into 1 row, combining the attributes
    # Add all of rows except case fatality ratio
    # Case fatality ratio = deaths / cases

    # Preserve keys
    province_state = grouped_df.iloc[0]["Province_State"]
    country_region = grouped_df.iloc[0]["Country_Region"]
    keys = {"Province_State": province_state, "Country_Region": country_region}

    # Condense rows
    temp_se = grouped_df[["Province_State",
                          "Country_Region",
                          "Confirmed",
                          "Deaths",
                          "Recovered",
                          "Active",
                          "Incidence_Rate"]].sum(min_count=1,
                                                 numeric_only=True)

    # Calculate case fatality
    deaths = temp_se["Deaths"]
    cases = temp_se["Confirmed"]
    if cases == 0:
        CF_ratio = 0
    else:
        CF_ratio = (deaths / cases) * 100
  
    # Build new row
    item = {"Case-Fatality_Ratio": CF_ratio}
    key_cols = pd.Series(data = keys)
    new_col = pd.Series(data = item)
    
    new_row = key_cols.append(temp_se)
    new_row = new_row.append(new_col)

    return new_row
'''
# ------------------
# 1.4 main
# ------------------

# Get location dataset
filename = 'dataset/location.csv'
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
result_filename = 'dataset/location_transformed.csv'
new_df.to_csv(result_filename, index=False)

# ========================================
# 1.5 JOINING DATASETS
# ========================================

# Get cases dataset
filename = 'dataset/cases_train.csv'
cases_csv = open(filename, 'rt')
cases_df = pd.read_csv(filename)
cases_csv.close()

# Get transformed location dataset
filename = 'dataset/location_transformed.csv'
loc_csv = open(filename, 'rt')
loc_df = pd.read_csv(filename)
loc_csv.close()

#print("before join:")
#print(cases_df.iloc[0])

joint_df = pd.merge(cases_df, loc_df, how="left")

#print("===========================================================================")
#print("after join:")
#print(joint_df.iloc[0])


# writing to CSV
result_filename = 'dataset/cases_joined.csv'
joint_df.to_csv(result_filename, index=False)

print("program end")

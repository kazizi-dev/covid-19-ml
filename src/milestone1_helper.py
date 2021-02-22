import pandas as pd

# ------------------
# 1.4 functions
# ------------------

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

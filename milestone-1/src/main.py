import os
import pandas as pd

from preprocessing import clean_age, clean_sex, clean_date, clean_cols, \
    drop_unused_columns, handle_skewed_data, transform_and_join

from visualization import plot_age_and_sex, plot_sex_distribution, \
    plot_outcome_distribution, plot_case_trend_per_outcome, \
    plot_world_map_cases, plot_countries_with_most_deaths


def clean_dataset(read_path, write_path):
    df = pd.read_csv(read_path)
    print(f'---> null values for {read_path}:')
    print(df.isnull().sum())

    clean_sex(df)
    clean_age(df)
    clean_date(df)
    # handle_skewed_data(df)    # TODO
    drop_unused_columns(df)
    clean_cols(df, ["country", "province"])

    df.to_csv(write_path, index=False)
    
    return df


if __name__  == '__main__':
    os.chdir(os.getcwd())

    train_df = clean_dataset('../data/cases_train.csv', 
                            '../results/cases_train_processed.csv')
    
    copy_df = train_df.copy()
    plot_age_and_sex(copy_df)
    plot_sex_distribution(copy_df)
    plot_outcome_distribution(copy_df)
    plot_case_trend_per_outcome(copy_df)

    test_df = clean_dataset('../data/cases_test.csv',
                            '../results/cases_test_processed.csv')


    # TODO
    # path = '../data/location.csv'
    # location_df = pd.read_csv(path)[["Province_State",
    #                             "Country_Region",
    #                             "Confirmed",
    #                             "Deaths",
    #                             "Recovered",
    #                             "Active",
    #                             "Incidence_Rate",
    #                             "Case-Fatality_Ratio"]]

    # print(f'---> null values for {path}:')
    # print(location_df.isnull().sum())

    # copy_df = location_df.copy()
    # plot_world_map_cases(copy_df)
    # plot_countries_with_most_deaths(copy_df)
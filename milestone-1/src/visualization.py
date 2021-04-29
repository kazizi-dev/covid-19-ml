import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
The following code is responsible for various plots
'''

def plot_age_and_sex(df):
    '''
    plot age distribution for male and female
    '''

    color_idx = 0
    colors = ['orange', 'red', 'blue']
    for sex in df['sex'].unique():

        sns.histplot(df['age'][df['sex'] == sex], color = colors[color_idx], 
                    label = 'Age', alpha=0.5, linewidth=1)

        plt.rcParams['figure.figsize'] = (10, 5)
        plt.title(f'Age Distribution for {sex}', size = 12)
        plt.ylabel('Count', size = 12)
        plt.xlabel('Age', size = 12)
        plt.savefig(f'../plots/age-distribution-for-{sex}-histogram.png')
        plt.close()

        color_idx += 1

def plot_sex_distribution(df):
    '''
    plot the distribution of each sex using a pie chart
    '''

    counts = list(df.sex.value_counts())
    labels = df.sex.value_counts()
    labels = ['unknown', 'male', 'female']
    colors = ['lightgrey', 'lightblue', 'lightpink']
    
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', 
            shadow=True, startangle=100)
    plt.title('Sex Distribution', size = 12)
    plt.savefig('../plots/sex-distribution-pie-chart.png')
    plt.close()


def plot_outcome_distribution(df):
    '''
    graph the percentage of each outcome in a pie chart
    '''

    counts = list(df['outcome'].value_counts())
    labels = df['outcome'].value_counts()
    labels = df['outcome'].unique().tolist()

    colors = ['lightgreen', 'lightpink', 'lightblue', 'red']
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', 
            shadow=True, startangle=100)
    plt.title('Outcome Percentage', size = 12)
    plt.savefig('../plots/outcome-distribution-pie-chart.png')
    plt.close()


def plot_case_trend_per_outcome(df):
    '''
    plot time series trend per outcome 
    '''

    colors = ["green", "purple", "orange", "red"]
    customPalette = sns.set_palette(sns.color_palette(colors))
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'])

    sns.lineplot(x='date_confirmation', y=range(len(df['outcome'])), 
                data=df, hue='outcome')
    plt.title('Case Trend Per Outcome', size = 12)
    plt.ylabel('Total Cases')
    plt.xlabel('Date')
    plt.savefig('../plots/time-series-trend-per-outcome.png')
    plt.close()


def plot_world_map_cases(df):
    '''
    plot covid cases based on logitude and latitude
    '''

    df = df.filter(['Lat', 'Long_'], axis=1)
    df = df.dropna()

    plt.scatter(df['Long_'], df['Lat'], marker='o', c='k', alpha=0.5, s=4)
    plt.title('Confirmed cases based on Latitude/Longitude')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.savefig('../plots/covid-cases-world-map.png')
    plt.close()


def plot_countries_with_most_deaths(df):
    '''
    plot a horizontal bar graph showing top 10
    '''

    df = df.groupby(by='Country_Region').mean()
    df = df.sort_values(by='Deaths', ascending=True).tail(10).reset_index()

    plt.barh(y=df['Country_Region'], width=df['Deaths'], color='brown')
    plt.title('Top 10 countries with most deceased cases')
    plt.xlabel('Total Cases')
    plt.ylabel('Countries')
    plt.savefig('../plots/top-10-countries-with-most-death.png')
    plt.close()
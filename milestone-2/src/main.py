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
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ 2.2 RANDOM FORESTS ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~ Get data ~~~~
filename = 'data/cases_joined.csv'
training = open(filename, 'rt')
csv_reader = csv.reader(training, dialect = 'excel')
training_df = pd.read_csv(filename)
training.close()

# ~~~~ Split data and labels ~~~~
# dropped all location columns except for country
# to minimize use of one hot encoding
training_labels = training_df[['outcome']].values
training_labels = training_labels.ravel()
training_data = training_df.drop(['outcome',
                                  'date_confirmation',
                                  'longitude',
                                  'latitude',
                                  'province'], axis=1)

# ~~~~ One hot encoding ~~~~
one_hot_enc = OneHotEncoder()
categorical_data = training_data[['sex', 'country']]

binary_data = one_hot_enc.fit_transform(categorical_data).toarray()
binary_labels = np.append(one_hot_enc.categories_[0], one_hot_enc.categories_[1])

encoded_df = pd.DataFrame(binary_data, columns=binary_labels)

# numerical data
numerical_df = training_data.drop(['sex', 'country'], axis=1)

# append converted data and numerical data
numerical_df = numerical_df.join(encoded_df)

# fill NaN values
print("...filling NaN values")
filled_encoded_df = numerical_df.fillna(value=0)

# ~~~~ Classifier ~~~~
print("...building model")
rand_forest_clf = RandomForestClassifier()

model = rand_forest_clf.fit(filled_encoded_df, training_labels)

# save model
print("...saving model")
pkl_filename = "./models/rf_classifier.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


'''
# writing to CSV
filled_encoded_df = filled_encoded_df.join(training_df[['outcome']])
print("...writing to csv")
result_filename = 'data/cases_joined_converted.csv'
#numerical_df.to_csv(result_filename, index=False)
filled_encoded_df.to_csv(result_filename, index=False)
'''

filled_encoded_df = filled_encoded_df.join(training_df[['outcome']])

# Load model
print("...loading model")
pkl_filename = "rf_classifier.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Use model
print("...using model on data")
Xdata = filled_encoded_df.drop(['outcome'], axis=1)
Ydata = filled_encoded_df[['outcome']]

# print score
score = pickle_model.score(Xdata, Ydata)
print("Train score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(Xdata)

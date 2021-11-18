# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:46:56 2021

@author: SKLAVOUG
"""

import pandas as pd
import json
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import sys

pd.options.mode.chained_assignment = None 

def runthrough(df):
    
    # Convert release date to a day of the year, based on the idea that movies released
    # around a particular time (e.g., Christmas, holidays) earn more revenue.
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_date'] = df['release_date'].apply(lambda row: row.timetuple().tm_yday)
    
    # Find films w budget or revenue <1000 and remove them, as they're bad data (given
    # it's unlikely a film can make/have a budget of less than $1000). Another option
    # was to just fix these records, i.e., multiply anything less than 1000 by 1 million,
    # but this caused issues with low-budget films which were sleeper hits, so I just
    # decided to remove them from the dataset instead.
    df.loc[train['revenue'] < 1000,'revenue'] = np.nan
    df.loc[train['budget'] < 1000, 'budget'] = np.nan
    
    df.dropna(subset=['revenue'], inplace=True)
    df.dropna(subset=['budget'], inplace=True)
    df.dropna(subset=['rating'], inplace=True)
    
    return df


def run(train, col, training, regression):
    
    # Extract value from a JSON field and append movie_id and/or budget
    # to each value, so it can be joined with the main dataframe after
    # the process is complete.
    def extract(item, col_name, json_list):
        idx = item['movie_id']
        if regression == True:
            idy = item['budget']
        item = item[col_name]
        item = json.loads(item)
        
        for line in item:
            line['movie_id'] = idx
            if regression == True:
                line['budget'] = idy
            json_list.append(line)
    
    # For the column in question, expand JSON data into its own dataframe
    json_list = []
    train.apply(lambda row: extract(row, col, json_list), axis=1)
    train.drop(col, inplace=True, axis=1)
    
    temp = pd.DataFrame(json_list)
    
    # Set a column of all true values, then drop all duplicates
    temp['vals'] = 1
    
    temp.drop_duplicates(inplace=True)        
    
    # Take only a subset of the 'crew' column, and conver the spoken_languages
    # field so it's in line with the others
    if col == 'crew':
        temp.drop(columns=['credit_id','gender','department','name'], inplace=True)
        temp = temp.loc[(temp['job'] == 'Producer')
                        |
                        (temp['job'] == 'Director')
                        |
                        (temp['job'] == 'Writer')]
        temp.drop(columns=['job'], inplace=True)

    if col == 'spoken_languages':
        temp.rename(columns={'iso_639_1': 'id'}, inplace=True)
        
    temp.drop_duplicates(inplace=True)
    
    # This probably comes close to overfitting, but I think I'm still clear
    # of it. They're different values for regression and classification based
    # on experimentation, but they also both seem reasonable. Note that
    # large values for cast in either regression or classification hurt the
    # result.
    
    if regression == True:
        # Sort by the highest-budget totals for each category.
        # Note that for classification this will need to change, and tbh doesn't
        # matter (for the classifier it could probably use much higher numbers)
        x = temp.groupby('id')['budget'].mean()
        
        x.sort_values(ascending=False, inplace=True)
        
        # Note that we're only taking the top values if training is true
        # , i.e., if it's the training data. If it's the test/validation
        # data then all columns will be taken, but those which aren't in
        # the training dataset will be removed and those not in the
        # test dataset will be added, so in the end they should be
        # exactly the same in terms of columns
        if training == True:
            if col == 'spoken_languages':
                x = x.head(10)
            elif col == 'cast':
                x = x.head(300)
            elif col == 'crew':
                x = x.head(200)
            elif col == 'keywords':
                x = x.head(20)
            elif col == 'production_companies':
                x = x.head(3)
            else:
                x = x.head(15)
    else:
        x = temp.groupby('id')['movie_id'].count()
        
        x.sort_values(ascending=False, inplace=True)
        
        if training == True:
            if col == 'spoken_languages':
                x = x.head(3)
            elif col == 'cast':
                x = x.head(50)
            elif col == 'crew':
                x = x.head(100)
            else:
                x = x.head(30)
    
    # Take only the top values from the temp df, then drop their duplicates
    # and convert to columns where '1' in a row means that id does feature
    # in that movie, and '0' means it doesn't.
    y = temp.loc[temp['id'].isin(x.index)]
    
    y.drop_duplicates(subset=['id','movie_id'])
    
    temp2 = y.groupby(['movie_id','id'])['vals'].first().unstack()
    
    # Append an identifier to the id, so id's can't be duplicated
    # across categories (e.g., same ID in both cast and crew JSONs)
    lis = []
    for i in list(temp2.columns):
        lis.append(str(i) + f'_{col}')
    temp2.columns = lis
    
    temp2.fillna(0, inplace=True)
    
    # Merge with the main df   
    merged = pd.merge(train, temp2, on='movie_id', how='left')
    merged.fillna(0, inplace=True)
    
    return merged

def regression(train, val):
    
    # Take only a subset of the columns
    cols = ['movie_id','budget','revenue','cast','runtime',
            'crew','genres','keywords','production_companies',
            'release_date','spoken_languages']
    
    train = train[cols]
    val = val[cols]
    
    # Split out the prediction value (revenue)
    x_train = train.drop(['revenue'], axis=1)
    y_train = train['revenue']
    x_test = val.drop(['revenue'], axis=1)
    y_test = val['revenue']
    
    # Split out JSON columns in both training and test taking the most common
    # x of each.    
    for i in ['keywords','cast','crew','genres','production_companies','spoken_languages']:
        if i in train.columns and i in val.columns:
            x_train = run(x_train, i, True, True)
            x_test = run(x_test, i, False, True)
    
    # Can't use any new columns in x_test. After run x_test keeps all its IDs, so
    # here we add any columns in x_train that aren't in x_test (as all 0s because
    # they're not present) and keep any columns that are in both, then drop any
    # columns that are in x_test but not x_train.
    removals = []
    
    for i in x_train.columns:
        if i not in x_test.columns:
            x_test[i] = 0
    
    for i in x_test.columns:
        if i not in x_train.columns:
            removals.append(i)
            
    x_test.drop(columns=removals, inplace=True, axis=1)
    
    # Define the regression model and fit it to the training data, then
    # predict the outcome
    model = GradientBoostingRegressor(max_depth=4)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    summ = pd.DataFrame(data={'id': [''],
                            'MSE': [round(mean_squared_error(y_test,y_pred),2)],
                            'correlation': [f'{round(pearsonr(y_test,y_pred)[0],2)}']})
    
    summ.to_csv('PART1.summary.csv', index=False)
    
    # Full output
    output = pd.DataFrame(data={'movie_id': x_test['movie_id'],
                                'predicted_revenue': y_pred})
    
    output = output.astype({'predicted_revenue': int})
    output.sort_values(by=['movie_id'], inplace=True)
    
    output.to_csv('PART1.output.csv', index=False)
    
def classification(train, val):
    
    # Take only a subset of columns in both training and test
    cols = ['movie_id','budget','runtime','rating','cast',
            'crew','genres','keywords',
            'release_date','spoken_languages']
    
    train = train[cols]
    val = val[cols]
    
    # Split out the prediction value (rating)
    x_train = train.drop(['rating'], axis=1)
    y_train = train['rating']
    x_test = val.drop(['rating'], axis=1)
    y_test = val['rating']
    
    # Split out JSON columns in both training and test taking the most common
    # x of each.
    for i in ['keywords','cast','crew','genres','spoken_languages']:
        if i in train.columns and i in val.columns:
            x_train = run(x_train, i, True, False)
            x_test = run(x_test, i, False, False)
    
    # Can't use any new columns in x_test. After run x_test keeps all its IDs, so
    # here we add any columns in x_train that aren't in x_test (as all 0s because
    # they're not present) and keep any columns that are in both, then drop any
    # columns that are in x_test but not x_train.
    removals = []
    
    for i in x_train.columns:
        if i not in x_test.columns:
            x_test[i] = 0
    
    for i in x_test.columns:
        if i not in x_train.columns:
            removals.append(i)
            
    x_test.drop(columns=removals, inplace=True, axis=1)
    
    # Define the classification model and fit it to the training data
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    
    # Predict the output based on the test data (minus the rating field)
    preds = model.predict(x_test)
    
    # Get macro average of precision and recall and overall accuracy, rounded to 2
    # decimal places
    precision = round(precision_score(y_test, preds, average='macro'),2)
    recall = round(recall_score(y_test, preds, average='macro'),2)
    accuracy = round(accuracy_score(y_test, preds),2)
    
    # Summary
    summ = pd.DataFrame(data={'id': [''],
                              'average_precision': [precision],
                              'average_recall': [recall],
                              'accuracy': [accuracy]})
    
    summ.to_csv('PART2.summary.csv', index=False)
    
    # Full output
    output = pd.DataFrame(data={'movie_id': x_test['movie_id'],
                                'predicted_rating': preds})
    
    output.sort_values(by=['movie_id'])
    
    output.to_csv('PART2.output.csv', index=False)

if __name__ == '__main__':
    # Open csv files
    train = pd.read_csv('training.csv')
    val = pd.read_csv('validation.csv')
    
    # Basic preprocessing including removing invalid values for budget
    # and revenue and converting the release date to a day of the year
    train = runthrough(train)
    val = runthrough(val)
    
    # Shuffle both sets
    train = shuffle(train)
    val = shuffle(val)
    
    # Run regression and classification on copies of the sets
    regression(train.copy(), val.copy())
    classification(train.copy(), val.copy())
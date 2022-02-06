# Importing libraries for Data Handling and EDA
import pandas as pd
import numpy as np

def prioritize_and_drop_redundant_columns(df, correlation_threshold, top_item_limit):
    ''''
    INPUTS:
    df - (DataFrame) with one hot encoded strings in which highly correlated strings will be dropped
    correlation_threshold - (float) from which all strings with higher Pearson's correlation will be dropped
    top_item_limit - limits the dataframe to the top (int) strings
    OUTPUTS:
    df - (DataFrame) with the dropped columns
    '''

    # Counting and ordering in descending order the most recurrent strings in the dataset 
    item_counting = (((df > 0).sum(axis = 0)/df.shape[0]).sort_values(ascending = False))
    
    # Limiting the DataFrame to its most recurrent items
    column_focus = item_counting.index[0:top_item_limit]
    df = df[column_focus]

    # Calculating correlation
    correlation_test = df.corr(method = 'spearman')
    
    # Storing highly correlated string columns
    redundant_columns = []

    # Looping rows from the dataset
    for row, series in enumerate(correlation_test[1:len(correlation_test)]):
        # Looping the columns in the current Series of the current row
        for column in range(0, row):

            if correlation_test.iloc[row, column] > correlation_threshold and row != column:
                redundant_columns.append(correlation_test.columns[column])

    redundant_columns = list(set(redundant_columns))

    df = df.drop(redundant_columns, axis = 1)

    return df

def above_median_metric(df):
    '''
    INPUTS:
    df - (DataFrame) with the columns neighbourhood_key and reviews_per_month
    OUTPUTS:
    df - (DataFrame) with a new column (above_median) stating 
    if the listing has a review_per_month above the median for its neighbourhood
    '''
    median_series = df.groupby('neighbourhood_key')['reviews_per_month'].median()

    above_median = []
    
    for index in df.index:
        if df.reviews_per_month[index] > median_series[df.neighbourhood_key[index]]:
            above_median.append(1)
        else:
            above_median.append(0)
            
    df['above_median'] = above_median

    return df
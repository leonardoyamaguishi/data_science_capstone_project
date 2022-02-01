# Importing libraries for Data Handling and EDA
import pandas as pd
import numpy as np
import scipy
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from apriori_python import apriori

# Plotting liberaries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def count_tokens(itemset):
    # Count the amount of most frequent tokens and token combinations
    token_count = 0

    for key in freqItemSet:
        token_count += len(freqItemSet[key])

    return token_count    

def list_relevant_columns(df, frequent_sets):
    '''
    INPUTS:
    df - (DataFrame) in which the most frequest items subsets (columns) will be listed
    frequent_sets - (dict) of (sets) of (frozensets) of items
    OUTPUTS:
    relevant_columns - (list) of the columns with the most common elements
    '''
    active_sets = []

    for item_set in frequent_sets[1]:
        for item in item_set:
            active_sets.append(item)
            
    relevant_columns = []

    for col in df.columns:
        if col in active_sets:
            relevant_columns.append(col)
            
    return relevant_columns

def ratio_of_true_values(df, relevant_columns):
    '''
    INPUTS:
    df - (DataFrame) in which the true values (1) will be summed
    relevant_columns - (list) of columns with the relevant items
    DESCRIPTION:
    This function is to determine how active the columns filetred by the apriori algorithm are
    OUTPUTS:
    ratio_of_true_values - average of the ratio of true values per column
    '''
    # Given the definition of the apriori algorithm, only the single element subsets are enough
    # for this calculation
    ratio_of_true_values = (df[relevant_columns].sum(axis = 0)/df.shape[0]).mean()
    
    return ratio_of_true_values

def evaluate_apriori_sets(df, list_of_subsets, minsup, minconf):
    '''
    INPUTS:
    df - (DataFrame) in which the apriori algorithm will be applied
    list_of_subsets - (list) of item subsets (lists of strings)
    minsup - the minimum probability for an item subset to be considered popular
    minconf - the minimum probability for an item combination to have its populatity evaluated
    OUTPUTS: 
    freqItemSet - item sets that meet the minsup requirement
    activation_index - (float) the ratio of true values per columns in the dataset from the strings selected by the
    apriori algorithm
    '''
    freqItemSet, rules = apriori(list_of_subsets, minsup, minconf)
    
    relevant_columns = list_relevant_columns(df, freqItemSet)
    
    activation_index = ratio_of_true_values(df, relevant_columns)
    
    return freqItemSet, activation_index

def create_item_set(df):
    ''''
    INPUT:
    df - (DataFrame) with boolean columns representing different strings within a text
    OUTPUT:
    item_set - (list) of subsets (also lists) containing the strings for each row in the original dataset
    '''
    item_set = []

    # Loop rows in the dataset
    for row in range(0, len(df)):
        item_subset = []
        # For each column in the dataset, if the boolean is 1
        # Retrieves the column name
        for index, boolean in enumerate(df.iloc[row]):
            if boolean == 1:
                item_subset.append(df.columns[index])
            
        item_set.append(item_subset)
    
    return item_set

def apriori_data_processing(df, minsup, minconf):
    '''
    INPUTS:
    df - (DataFrame) in which the apriori algorithm will be applied
    minsup - the minimum probability for an item subset to be considered popular
    minconf - the minimum probability for an item combination to have its populatity evaluated
    
    OUTPUTS:
    filtered_df - (DataFrame) df with the columns selected by the apriori algorithm filtered
    '''
    
    list_of_subsets = create_item_set(df)
    freqItemSet, rules = apriori(list_of_subsets, minsup, minconf)
    relevant_columns = list_relevant_columns(freqItemSet)
    
    filtered_df = df[relevant_columns]
    return filtered_df
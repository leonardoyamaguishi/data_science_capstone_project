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

def list_relevant_columns(frequent_sets):
    '''
    INPUTS:
    frequent_sets - (dict) of (sets) of (frozensets) of items
    OUTPUTS:
    relevant_columns - (list) of the columns with the most common elements
    '''
    relevant_columns = []

    for item_set in frequent_sets[1]:
        for item in item_set:
            relevant_columns.append(item)
            
    return relevant_columns

def ratio_of_true_values_single_values(df, freqItemSet):
    '''
    INPUTS:
    freqItemSet - item sets that meet the minsup requirement
    relevant_columns - (list) of columns with the relevant single values
    DESCRIPTION:
    This function is to determine how active the columns filetred by the apriori algorithm are
    OUTPUTS:
    ratio_of_true_values - (float) average of the ratio of true values per column
    '''
    # Given the definition of the apriori algorithm, only the single element subsets are enough
    # for this calculation

    relevant_columns = list_relevant_columns(freqItemSet)

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
    activation_index_single_items - (float) the ratio of true values per columns in the dataset from the strings selected by the
    apriori algorithm
    activation_index_pairs - (float) average of the ratio of true values per column
    '''
    freqItemSet, rules = apriori(list_of_subsets, minsup, minconf)
   
    activation_index_single_items = ratio_of_true_values_single_values(df, freqItemSet)
    
    activation_index_pairs = ratio_of_true_values_pair(df,freqItemSet)

    return freqItemSet, activation_index_single_items, activation_index_pairs

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

def list_relevant_pairs(frequent_sets):
    '''
    INPUTS:
    frequent_sets - (dict) of (sets) of (frozensets) of items
    OUTPUTS:
    relevant_pairs - (list) of the columns with the most common elements
    '''
    
    pair_list = []

    for subset in frequent_sets[2]:

        pair = []
        
        for item in subset:
            pair.append(item)
        
        pair_list.append(pair)

    return pair_list

def ratio_of_true_values_pair(df, frequent_sets):
    ''''
    INPUTS:
    df - (DataFrame) in which the most frequent active pairs will be measured
    frequent_sets - (dict) in which the pairs (key = 2) will be evaluated
    OUTPUTS:
    ratio_of_true_values - (float) average of the ratio of true values per column
    '''
    try: 

        pair_list = list_relevant_pairs(frequent_sets)

        activation_ratios = []

        for pair in pair_list:
            activation_ratios.append((df[pair].sum(axis = 1) == 2).sum()/df.shape[0])

        activation_ratios = pd.Series(activation_ratios)

        ratio_of_true_values = activation_ratios.mean()
        
        return ratio_of_true_values

    except:

        return 0

def plot_activation_index(tested_values, single_item_activation_list, item_pair_activation_list):
    ''''
    INPUTS:
    tested_values - (list) of iterated values (x) for the plot
    single_item_activation_list - (list) list of the activation ratios for single items
    item_pair_activation_list - (list) list of the activation ratios for item pairs
    OUTPUTS:
    None
    '''

    plt = sns.lineplot()
    plt.plot(tested_values, single_item_activation_list, color='r', label= 'single item activation')
    plt.plot(tested_values, item_pair_activation_list, color='b', label= 'item pair activation')
    plt.legend()

    return 
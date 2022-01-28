# Importing relevant libraries for EDA
import pandas as pd
import numpy as np
import seaborn as sns
import os
import scipy
import re
import ast

# Modules for data processing
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from datetime import datetime

import haversine as hs

'''
Functions for neighbourhood_hdi data processing
'''
def unpack_double_assessment(neighbourhood_hdi):
    '''
    INPUT:
    neighbourhood_hdi - (DataFrame) in which more than one neighbourhood was assessed at once
    Note: In such cases, the different neighbourhoods were listed in the same row, separated by ","
    
    OUTPUT:
    unpacked_neighbourhood_hdi - (DataFrame) with unpacked neighbourhoods (the rankings will be repeated)
    '''
    delist_after_treatment = []
    
    # Loops through indexes and neighbourhood names
    for index, neighbourhood in enumerate(neighbourhood_hdi.neighbourhood):
        
        # If the neighbourhood column has ','
        # means that more than one neighbourhood was assessed for the respective row
        if neighbourhood_hdi['neighbourhood'][index].find(',') != -1:
            
            # Lists indexes to drop after unpacking
            delist_after_treatment.append(index)
            
            # Lists the names inside the neighbourhood name column
            names_to_unpack = neighbourhood_hdi.neighbourhood[index].split(',')
            
            # Copies the entire row as a template, as the only change in the unpacked values will be the name
            template_series = neighbourhood_hdi.iloc[index]

            # Loops through the names to unpack
            for name in names_to_unpack:
                # Duplicates the template series
                series_to_append = template_series
                # Changes the neighbourhood name from the template series
                series_to_append['neighbourhood'] = name
                # Appends the new unpacked neighbourhood
                neighbourhood_hdi = neighbourhood_hdi.append(series_to_append, ignore_index = True)
    
    # After all iterations, the listed indexes to drop are dropped
    unpacked_neighbourhood_hdi = neighbourhood_hdi.drop(delist_after_treatment, axis = 0)
    
    return unpacked_neighbourhood_hdi

def neighbourhood_hdi_update_keys(neighbourhood_hdi):
    '''
    INPUTS:
    neighbourhood_hdi - (DataFrame) to have the keys created
    
    OUTPUTS:
    neighbourhood_hdi - (DataFrame) with updated keys according to their HDI rank
    '''
    neighbourhood_hdi.sort_values(by = 'human_development_index', ascending = False, inplace = True)
    neighbourhood_hdi['neighbourhood_key'] = list(neighbourhood_hdi.index)
    
    return neighbourhood_hdi

'''
Functions for listings data processing
'''
def tokenize(text):
    '''
    INPUTS:
    text - a string to be tokenized
    OUTPUTS:
    clean_tokens - (list) of clean tokens
    '''
    
    if text.isnumeric() == True:
        return ''
    else:
        # Removing stop words 
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

        # Tokenizing the text
        tokenized_text = word_tokenize(text)

        # Removing stopwords - assuming that all amenities are listed in english
        tokenized_text = [w for w in tokenized_text if w not in stopwords.words("english")]

        # If the text is not a stop word, it should be tokenized
        if tokenized_text != []:
            # Starts the lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Creates a list of lemmatized tokens
            clean_tok = lemmatizer.lemmatize(tokenized_text[0]).lower().strip()
        else:
            clean_tok = ''

        return clean_tok

def token_list_to_string(token_list):
    '''
    INPUTS:
    token_list - (list) of tokens
    
    OUTPUTS:
    token_list_string - (str) in which the tokens are separated by ' '
    '''
    token_list_string = ''
    
    n = 0
    
    for n in range (0, len(token_list)):
        
        # On the start of the loop, no space is necessary
        if n != 0:
            token_list_string += ' '
            
        # adds on the token_list_string the a unique token (can be a compound noun)
        token_list_string += token_list[n]
        
    return token_list_string

def literal_to_list_and_tokenize(list_string):
    '''
    INPUTS:
    listings_series - (Series) with a column with literal list strings composed of strings to be tokenized
    
    OUTPUTS:
    token_list_as_string - (str) in which the tokens are separated by ' ' (adapted for CountVectorizer)
    '''
    
    # Converts literal list to list
    converted_list = ast.literal_eval(list_string)
          
    for i in range (0, len(converted_list)):
        
        # As each item in the list is a item that could be a compound noun
        # The objective is to tokenize each element of the compound noun and join then with ' '
        # Increasing the chance of finding matches throughout the dataset
        
        compound_noun = converted_list[i].split()
        
        generated_word = ''
        
        for l in range(0, len(compound_noun)):
            tokenized_element = tokenize(str(compound_noun[l]).lower())
            
            # Tokenized_elemnt is '' when the given word is a stop word
            if tokenized_element != '':
                
                # If not a stop word, check if it's the first element of the possible compound noun
                if l != 0:
                    generated_word += ' '
                    
                # If compound_noun has length above 1, it's a compound noun
                generated_word += tokenize(str(compound_noun[l]).lower())
        
        # Adds the compound noun to the converted list
        converted_list[i] = generated_word

    # Convert the converted_list of tokenized compound nouns to a string separating them with ' '
    token_list_as_string = token_list_to_string(converted_list)
    
    return token_list_as_string #ASSESSMENT

def list_nan_columns(df):
    '''
    INPUT:
    df - (DataFrame) in which the columns composed entirely of NaN values will be listed
    OUTPUT:
    nan_columns_list - (list) of columns composed entirely of NaN values
    '''
    
    nan_columns_list = []
    
    for column in df.columns:
        if df[column].isna().sum() == df.shape[0]:
            nan_columns_list.append(column)
            
    return nan_columns_list

def compute_walking_dist_stations(listings, metro_stations, metro_stations_google_score, walking_dist = 1):
    '''
    INPUTS:
    listings - (DataFrame) with AirBnb listings with latitude and longitude columns
    metro_stations - (Dictionary) with the name of the metro stations as keys and (tuples)
    of their respective latitude and longitude
    
    OUTPUTS:
    nearest_stations_walking_dist - (list) of the nearest stations within a distance of 1 km
    '''
    
    # Prepares the lists
    nearest_stations_walking_dist = []
    
    # Loops through the latitudes and longitudes of the AirBnBs
    for latitude, longitude in zip(listings.latitude, listings.longitude):
        
        # Distance place holder
        nearest_station = 'NONE'
        nearest_station_dist = 999999
        
        # Loops through the metro_stations dictionary
        for station in metro_stations:
            
            # Computes distance according to the Haversine distance
            dist = hs.haversine((latitude, longitude), metro_stations[station])
            
            # Updates the variables if a nearer metro station is found
            if dist < nearest_station_dist:
                nearest_station = station
                nearest_station_dist = dist
        
        # Updates the lists
        if dist <= walking_dist:
            nearest_stations_walking_dist.append(station)
        else:
            nearest_stations_walking_dist.append('NONE')
        
        station_score_list = assign_station_score(nearest_stations_walking_dist, metro_stations_google_score)
    return nearest_stations_walking_dist, station_score_list

def listings_assign_keys(listings, neighbourhood_hdi):
    '''
    INPUTS: 
    listings - (DataFrame) with 'neighbourhood' column that matches the neighbourhoods in neighbourhood_hdi
    neighbourhood_hdi - (DataFrame) with 'neighbourhood' and 'neighbourhood_key' columns
    
    OUTPUTS:
    listings - (DataFrame) with new or updated 'neighrbouhood_key' column matching neighbourhood_hdi
    '''
    
    # Updating neighbourhood_hdi keys
    neighbourhood_hdi = neighbourhood_hdi_update_keys(neighbourhood_hdi)
    
    # Creates a dictionary that connects the neighbourhood name to its respective key
    name_key_dict = {}

    for name, key in zip(neighbourhood_hdi.neighbourhood, neighbourhood_hdi.neighbourhood_key):
        name_key_dict[name] = key

    listings_keys = []

    # Assigns the secondary key to listings 'neighbourhood_key' column according to its 'neighbourhood'
    for name in listings.neighbourhood:
        listings_keys.append(name_key_dict[name])

    listings['neighbourhood_key'] = listings_keys
    
    return listings

def assign_station_score(stations_list, metro_stations_google_score):
    '''
    INPUTS:
    stations_list - (list) of metro stations mapped in the dictionaries declared in the start of the notebook
    metro_stations_google_score - (dictionary) that links the metro stations (keys) with their respective
    google scores
    
    OUTPUTS:
    station_score_list - (list) of google scores in the same order as the stations_list

    '''
    
    station_score_list = []
    
    for station in stations_list:
        if station != "NONE":
            station_score_list.append(metro_stations_google_score[station])
        else: 
            # -1 is assigned to the case NONE
            station_score_list.append(-1)
            
    return station_score_list

def summarize_bathroom_string(listings):
    '''
    INPUTS:
    listings - DataFrame with bathrooms_text, in which "shared" indicated shared bathrooms
    
    OUTPUTS:
    shared_bathroom - (list) with 1 for True and 0 for False
    bathroom_qty - (list) quantity of bathroom for the respective listing
    '''
    shared_bathroom = []
    bathroom_qty = []

    for txt in listings.bathrooms_text:
        
        # Retrieving the bathroom quantity
        try:
            bathroom_qty.append(float(str(txt)[0:(str(txt).lower().find(' '))]))
        except:
            # Some strings do not have quantity, therefore are considered as nan
            bathroom_qty.append(np.nan)
            
        # If the bathrooms_text has shared in the string, it indicates a shared bathroom
        if str(txt).lower().find('shared') != -1:
            shared_bathroom.append(1)
        else:
            shared_bathroom.append(0)
            
    return bathroom_qty, shared_bathroom

def standardize_boolean_columns(txt):
    '''
    INPUTS:
    txt - (string or nan) with "t" for True and "f" for False
    OUTPUTS:
    1 - (int) for "t"
    0 - (int) for "f" and nan
    '''
    if txt == 't': # Case t (True)
        return 1
    else: # Cases f (False) and nan
        return 0

def compute_days_since_start(df, timecol_before, timecol_after, delta_col_name, drop_original):
    '''
    INPUTS:
    df - (DataFrame) with 2 columns with dates in string format (%Y-%m-%d)
    timecol_before - (string) older dates column name
    timecol_after -  (string) newer dates column name
    delta_col_name - (string) name of the column with the calculated time difference
    drop_original - (boolean) if the original time columns must be dropped
    
    OUTPUTS:
    df - (DataFrame) with a column named delta_col_name with the days between the two given date columns

    '''
    
    # Converting the dates in string format to timestamp
    df = time_converter(df, timecol_before)
    df = time_converter(df, timecol_after)
    
    # Creating a list to store the time subtractions
    days_since = []
    
    # Loops through the time columns 
    for date_before, date_after in zip(df[timecol_before], df[timecol_after]):
        
        # Calculates the difference if the elements are timestamps
        if type(date_before) == pd._libs.tslibs.timestamps.Timestamp and type(date_after) == pd._libs.tslibs.timestamps.Timestamp:
            days_since.append((date_after - date_before).days)
        
        # Else nan is appended
        else:
            days_since.append(np.nan)
    
    # Dropping the original time columns
    if drop_original == True:
        df.drop([timecol_before], axis = 1, inplace = True)
        df.drop([timecol_after], axis = 1, inplace = True)
    
    # Appending the difference column
    df[delta_col_name] = days_since
    
    return df
    
'''
Functions for general data processing
'''
def standardize_names(txt):
    '''
    INPUTS:
    txt - (string) in which the characters will be converted to lower case and the spaces will be removed
    OUTPUTS:
    standardized_txt - lower case (string) without spaces
    '''
    return str(txt.lower().replace(' ', ''))

def time_converter(df, time_str_column):
    """
    INPUT:
    df - DataFrame that contains a column with time strings, this column must nost contain nan values.
    time_str_column - name of the column with the time strings.
    
    This function converts the time strings in a given column using the strptime function from the datetime library.
    By doing this, a new column with the converted elements will be created and the old column will be dropped.
    
    OUTPUT:
    df - DataFrame with a new converted column with the "con_" prefix.
    
    """
    converted_time = []
    new_column_name = time_str_column   
    
    for time in df[time_str_column]:
        if type(time) == str:
            converted_time.append(datetime.strptime(str(time), "%Y-%m-%d"))
        else:
            converted_time.append(np.nan)
            
    df = df.drop(time_str_column, axis = 1)
    df[new_column_name] = converted_time
    
    return df
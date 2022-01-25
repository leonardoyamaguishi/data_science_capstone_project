def unpack_double_assessment(neighbourhood_hdi = neighbourhood_hdi):
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

def standardize_names(txt):
    '''
    INPUTS:
    txt - (string) in which the characters will be converted to lower case and the spaces will be removed
    OUTPUTS:
    standardized_txt - lower case (string) without spaces
    '''
    return str(txt.lower().replace(' ', ''))

def neighbourhood_hdi_update_keys(neighbourhood_hdi = neighbourhood_hdi):
    '''
    INPUTS:
    neighbourhood_hdi - (DataFrame) to have the keys created
    
    OUTPUTS:
    neighbourhood_hdi - (DataFrame) with updated keys according to their HDI rank
    '''
    neighbourhood_hdi.sort_values(by = 'human_development_index', ascending = False, inplace = True)
    neighbourhood_hdi['neighbourhood_key'] = list(neighbourhood_hdi.index)
    
    return neighbourhood_hdi
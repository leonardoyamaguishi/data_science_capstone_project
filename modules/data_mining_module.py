# Importing libraries for Data Handling and EDA
import pandas as pd
import numpy as np
import scipy

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_features(df, y_predicted, feature_importance_series):
    '''
    INPUTS:
    df - (DataFrame) in which a prediction was performed
    feature_importance_series - (Series) of feature importance ranked by estimator
    y_predicted - (array) of predicted classification outputs [0,1]
    
    DESCRIPTION:
    Orders the feature_importance_series in descending order and for each feature a histogram is plotted
    The hue of the histogram is the prediction output
    This function assists in the interpretation of how each feature contribute to the predicted output
    
    OUTPUTS: 
    none
    '''
    
    feature_importance_series = feature_importance_series.sort_values(ascending = False)
    
    for feature in feature_importance_series.index:
        fig = plt.figure(figsize = (8,8))
        plt.ylim(0,600)
        sns.histplot(x = df[feature], hue = y_predicted)
        
    return

def evaluate_classification(y_real, y_prediction):
    '''
    INPUTS: 
    y_real - (array) of output boolean [0, 1] original values from a given dataset 
    y_prediction - (array) of output boolean [0, 1] predicted values from a given dataset
    
    OUTPUTS:
    none
    
    Prints the Confusion Matrix based on the given inputs
    '''
    
    acc = accuracy_score(y_real, y_prediction)
    print('Model accuracy: {}'.format(acc))
    
    actual_balance = y_real.sum()/y_real.shape[0]
    print('Actual ratio of positive classifications: {}'.format(actual_balance))
    
    c_m = confusion_matrix(y_real, y_prediction)


    label_0_0 = "True Negative:\n " + str(round((100*c_m[0,0])/c_m.sum(),1)) + '%'
    label_0_1 = "False Positive:\n " + str(round((100*c_m[0,1])/c_m.sum(),1)) + '%'
    label_1_0 = "False Negative:\n " + str(round((100*c_m[1,0])/c_m.sum(),1)) + '%'
    label_1_1 = "True Positive:\n " + str(round((100*c_m[1,1])/c_m.sum(),1)) + '%'

    labels = [[label_0_0, label_0_1],[label_1_0, label_1_1]]

    fig = plt.figure(figsize = (8,8))
    sns.heatmap(c_m, annot = labels, fmt = '', cmap = 'Blues')
    
    return 

def n_estimator_cv_analysis(start, stop, step, estimator, X_train, y_train):
    '''
    INPUTS:
    start - (int), starting number of estimators for the iteration
    stop - (int), ending number of estimators for the iteration
    step - (int), step size between the iterations of number of estimators
    estimator - sklearn model in which n_estimators is one of the inputs parameters
    X_train - (nd.array) of inputs for the training dataset
    y_train - (nd.array) of outputs for the training dataset
    
    DESCRIPTION:
    Iterates between the given start and stop numbers to evaluate how the prediction scores vary
    in cross validation with the increase of estimators
    
    OUTPUTS:
    none
    '''
    n_estimator_count = []
    score_list = []
    score_type_list = []

    for n_estimators in range(start, stop, step):
        
        # Setting the n_estimators of the iteration
        param = {'n_estimators' : n_estimators}
        estimator.set_params(**param)

        cv_score_list, cv_score_type_list = cross_validation_score_lists(estimator,
                                                                         X_train,
                                                                         y_train)

        n_estimator_count = n_estimator_count + ([n_estimators]*10)
        score_list = score_list + cv_score_list
        score_type_list = score_type_list + cv_score_type_list

    score_df = pd.DataFrame({'n_estimators': n_estimator_count,
                             'score' : score_list,
                             'score_type' : score_type_list})

    # Plotting the results from the Random Forest Classifier
    sns.lineplot(x = 'n_estimators', y = 'score', hue = 'score_type', data = score_df, ci = 95)

def tree_cv_feature_importance(estimator, X_train, y_train, input_features):
    '''
    INPUTS:
    estimator - sklearn model in which cross validation will be performed
    X_train - (nd.array) of inputs for the training dataset
    y_train - (nd.array) of outputs for the training dataset
    input_features - (list) of input features in the same order as they are presented in 
    
    OUTPUTS: 
    feature_importance_series - (Series) of feature importance with features as indexes
    '''

    cv_results = cross_validate(estimator,
                                X_train,
                                y = y_train,
                                return_estimator = True,
                                return_train_score = True)


    feature_importance_df = []

    # Assessing each developed estimator
    for i in range(0,5):
        # Retrieving its feature importances
        feature_importance_df.append(cv_results['estimator'][i].feature_importances_)

    feature_importance_df = pd.DataFrame(feature_importance_df, columns = input_features)
    feature_importance_series = feature_importance_df.mean(axis = 0).sort_values(ascending = False)
    
    return feature_importance_series[feature_importance_series > 0]

def cross_validation_score_lists(model_for_cv, X_train, y_train):
    '''
    INPUTS:
    model_for_cf - sklearn estimator to be evaluated using 5-fold cross validation
    X_train - nd.array with the prediction data from the train dataset
    y_train - array with the output data from the train dataset

    OUTPUTS:
    cv_score_list - (list) of with 10 train and validation scores
    cv_score_type_list - (list) of score types, 5 for train and 5 for validation
    '''
    
    cv_results = cross_validate(model_for_cv,
                                X_train,
                                y = y_train,
                                return_estimator = True,
                                return_train_score = True)
    
    # By standard the cross validator performs a 5-fold cross validation
    
    # Storing the results in a DataFrame with the columns n_estimators, score and score_type
    # The idea is to be able to visualize the confidence interval of the scores
    # Inspiration: https://stackabuse.com/seaborn-line-plot-tutorial-and-examples/
    
    cv_score_list = []
    cv_score_type_list = []
    
    for i in range(0,5):
        cv_score_list.append(cv_results['train_score'][i])
        cv_score_type_list.append('train')

        
    for i in range(0,5):
        cv_score_list.append(cv_results['test_score'][i])
        cv_score_type_list.append('test')
    
    return cv_score_list, cv_score_type_list
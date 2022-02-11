# Udacity's Data Science Capstone Project: 
## Airbnb as a source of income, a customer focused Data Science analysis to optimise investments

In this repository, the process of Knowledge Discovery in Databases (KDD) is applied in [data scraped](http://insideairbnb.com/index.html) from [Airbnb](https://airbnb.com.br) listings from Rio de Janeiro (Brazil) in order to evaluate which features displayed in the listing's page could be highly related with superior booking counts.

The complete article created for this project is available in this [link](https://leonardo-yamaguishi.medium.com/airbnb-as-a-source-of-income-a-customer-focused-data-science-analysis-to-optimise-investments-d4a2d87bf627)

## Project libraries:
* [apriori-python](https://pypi.org/project/apriori-python/)
* [ast](https://docs.python.org/3/library/ast.html)
* [datetime](https://docs.python.org/3/library/datetime.html)
* [haversine](https://pypi.org/project/haversine/)
* [matplotlib](https://matplotlib.org)
* [Natural Language Toolkit](https://www.nltk.org)
* [NumPy](https://numpy.org)
* [os](https://docs.python.org/3/library/os.html)
* [pandas](https://pandas.pydata.org)
* [re](https://docs.python.org/3/library/re.html)
* [scikit-learn](https://scikit-learn.org)
* [SciPy](https://scipy.org)
* [Seaborn](https://seaborn.pydata.org)
* [sys](https://docs.python.org/3/library/sys.html)

## Files in this repository

**data:** the folder with the data analysed in this project and further information
* listings.csv: contains the Airbnb listigs data from Rio de Janeiro
* neighbourhood_hdi: contains socioeconomic indices per neighbourhood in Rio de Janeiro
* amenities_post_etl.csv: amenities dataset generated on the ETL phase of the project
* listings_post_etl.csv: listings dataset generated on the ETL phase of the project
* misc: folder that contains the Inside Airbnb Data Dictionary

**modules:** the folder with the created modules for the project
* airbnb_data_processing.py: script created for the ETL workflow
* apriori_module.py: script for the calculations involving the Apriori algorithm
* data_mining_module.py: script for the support functions used in the Data Mining phase of the project
* modelling_support.py: script for dataset handling pre Data Mining

**notebooks:** the folder with the Jupyter Notebooks used for this project
* exploratory_data_analysis.ipynb: notebook for the EDA phase
* etl_notebook.ipynb: notebook for the ETL phase
* data_mining.ipynb: notebook for the Data Mining phase

## Problem
Airbnb as a source of income, optimising investments by focusing on the tourists most considered features for booking.

## Solution proposal
Based on the given problem, the proposed approach is to develop a decision based estimator and, based on the recognised patterns, assess how its most relevant features are related to higher success rates.

## Summary of results: 
Although both trained Random Forest Classifiers performed above the defined baseline (50%), the model using the entire dataset provided relevant features with clearer correlations with the final outputs, indicating that the estimator probably had more standardised and unilateral decision patterns for the most relevant features. Interestingly, although the continuous variables weren't normally distributed, the model could identify subsets with different distributions, specially for the review scores.

On the other hand, the estimator that used only amenities as inputs had a more interesting potential for insight generation as the inputs were mostly possible material investments. Another relevant factor is that the features weren't absolutely clear in literal terms, something that could affect the modelling and interpretation phases negatively. Moreover, the chosen NLP approach was reasonably simplistic when compared to the problem of reducing lists of non standardised composed nouns to sufficiently generic interpretable features.

Regarding the added features such as socioeconomic indicators, the hypothesis that they could complement the analysis was discarded as a result of the chosen feature selection process.

## Aknowledgements:
* [Haversine explanation](https://towardsdatascience.com/calculating-distance-between-two-geolocations-in-python-26ad3afe287b)
* [Apriori explanation](https://towardsdatascience.com/apriori-association-rule-mining-explanation-and-python-implementation-290b42afdfc6)
# Football Match Prediction

###### *For a more detailed description of this work, please read the thesis.pdf file. However, please note that there may be some minor differences between the results presented in the thesis.pdf file and those obtained using the provided data and code. These differences are due to some minor changes that were made to the data preparation processes after the paper was written.*

## Table of Contents

- [Abstract](#abstract)
- [Data Sources](#datasources)
- [Methodology](#methodology)
- [Classification Models](#classificationmodels)
- [Results](#results)
- [Conclusions](#conclusions)


<a id='abstract'></a>
## Abstract

This work explores using Machine Learning to predict football match outcomes in the top five European leagues from season 2016/2017 to 2021/2022. The study aims to expand on previous literature by analyzing a more extensive range of football-related features and assessing the predictive power of different football knowledge forms to understand  better  the  various  components  of  match  information  and  the  overall application domain. Hence, the analysis was conducted on two levels: first, splitting the data into seven subsets, each containing a reduced feature space, and then aggregating them into a comprehensive dataset. Several combinations of classification algorithms and preprocessing  techniques,  such  as  categorical  encoding,  feature  scaling,  and dimensionality  reduction,  were  tested  on  each  set  in  a  structured  way,  enabling comparisons.  The models trained on different sub-datasets achieved varying levels of test accuracy, ranging from 48.7% for the Venue set, 53.9% for the Standings set, 54.5% for the Form and Rest set, 52.7% for the Stats set, 60.0% for the Betting Odds set, 54.1% for the Team Attributes set, and 54.2% for the Player Attributes set. In comparison, the model trained on the Comprehensive dataset accomplished a test accuracy of 60.7%. The process of generating  such  results  provides  critical  insights  for  further  research  on  practical applications by uncovering the predictive ability of the diverse feature spaces. Moreover, based on the comprehensive dataset, the best-performing model performs satisfactorily, especially compared to prior multiclass analyses. Its potential practical utilization in future studies could be of interest, mainly to investigate its effectiveness in identifying betting strategies. 



<a id='datasources'></a>
## Data Sources
Our study collected and analyzed football match data from the top five European leagues (Bundesliga, La Liga, Ligue 1, Premier League, and Serie A) from the 2016/2017 season to the 2021/2022 season. The scope of our research did not include games from domestic and European cups. However,  observations from these competitions were nevertheless considered to create several features employed in the study.  

The data was gathered from three online sources: 
- [Sportmonks football API](https://docs.sportmonks.com/football/)
- [FIFA index website](https://www.fifaindex.com/)
- [Visual Crossing Weather API](https://www.visualcrossing.com/weather-api) 

From the Sportmonks football API, we acquired detailed information about football fixtures, including game results, statistics, betting odds, in-game events, and starting lineup players. 
The FIFA index website contains details from the FIFA video games series referring to both teams' and players' attributes. The information from this source was extracted through web scraping, a technique to collect large amounts of internet data automatically. The scraped data was organized into two separate databases: one for teams with approximately 82,000 observations and one for players with roughly 1,250,000 instances. 
From the Visual Crossing Weather API, we attained meteorological variables referring to the weather conditions for the matches considered in our analysis. 

A total of 10,536 matches were considered in this analysis. 
![data_sources_plot](https://user-images.githubusercontent.com/80990030/228912418-852a7749-d4d3-48cb-8a37-c23b9834400c.png)



<a id='methodology'></a>
## Methodology

Our study addressed the issue of predicting football outcomes exclusively using a **classification** framework since most research on football prediction has implemented classification practices. Furthermore, we decided to utilize a **multiclass classification** approach rather than the binary approach implemented in a few previous studies. This decision was based on the intrinsic nature of football, which has a high probability of draws—considering such a category increased the learning task's complexity and caused a decline in accuracy, especially since draws are the most challenging class to predict. Also, we defined **accuracy** as the performance metric to evaluate models. This decision was based on the fact that accuracy is employed in most research studies within the specific application domain. 

The literature on football match prediction has been afflicted by the inability to compare approaches from different papers due to the focus on different samples, caused primarily by the absence of a shared and comprehensive dataset. The division of our entire data into lower-dimensional feature spaces attempted to address this issue by comparing between sub-datasets. By reducing the initial dimensionality of the data, this approach facilitated the management of existing data and the development of new attributes while also providing a deeper understanding of different aspects of football match information. Notably, each sub-dataset was arbitrarily partitioned to contain knowledge referring to a specific field, differing in the nature and 
design of the included information or sharing a particular affinity.
Another crucial novelty of our approach derives from utilizing a vast and comprehensive dataset. While previous studies have, in a few cases, used datasets of larger size in terms of the number of matches than our dataset, this study is novel in considering a more extensive feature space. This analysis significantly increased the data dimensionality by almost an order of magnitude compared to prior studies. 

A shared set of techniques was  employed throughout this study to ensure comparability across the results obtained from different datasets, including all the subsets and the comprehensive dataset. 
The first step of our standard approach was an 80/20 split of the available data into separate training and testing sets through stratified sampling.
After the split and to conduct an exhaustive analysis, we employed a systematic procedure to identify the optimal model while improving its robustness and generalization. This approach involved a grid search, with five-fold cross-validation for evaluation, over various preprocessing techniques and classification  algorithms. Additionally, hyperparameter tuning was  applied for each model to enhance its performance. 
The grid search considers various preprocessing techniques, such as categorical encoding (one-hot and ordinal) and feature scaling (normalization  and  standardization). Also, dimensionality reduction methods were assessed, including feature extraction (PCA) and feature selection processes (mutual 
information, F-test, decision tree-based, and random forest-based). Moreover, the grid search also evaluates several classifiers and respective hyperparameter tuning, including Naive Bayes, kNearest Neighbors (k-NN), Multinomial Logistic Regression, Decision Tree, Random Forest, and AdaBoost.

Data cleaning and feature engineering methods are described separately for each dataset.



<a id='results'></a>
## Results

**Venue Sub-Dataset**: Random Forest classifier achieved the best test accuracy of **48.7%** when combined with ordinal encoding for categorical variables, normalization for feature scaling, and Mutual Information feature selection.
**Standings Sub-Dataset**: Multinomial Logistic regression classifier achieved the best test accuracy of **53.9%** when combined with normalization for feature scaling and F-test feature selection.***
**Form and Rest Sub-Dataset**: k-Nearest Neighbors classifier achieved the highest test accuracy of **54.5%**, combined with normalization for feature scaling and random forest-based feature selection.***
**Stats Sub-Dataset**: Multinomial Logistic regression classifier achieved the best test accuracy of **52.7%**, combined with standardization for feature scaling and F-test feature selection (no nominal attributes were present in the feature set, eliminating the need for any categorical encoding technique in the grid search). ***
**Odds Sub-Dataset**: k-Nearest Neighbors classifier achieved the highest test accuracy of **60.0%**, combined with standardization for feature scaling and random forest-based feature selection. *** 
**Team Attributes Sub-Dataset**: k-Nearest Neighbors classifier achieved the best test accuracy of **54.1%** when combined with One-Hot encoding for categorical variables, standardization for feature scaling, and F-test feature selection. 
**Player Attributes Sub-Dataset**: Multinomial Logistic regression classifier achieved the best test accuracy of **54.2%**, combined with One-Hot encoding for categorical variables, standardization for feature scaling, and no feature selection technique since a better performance was accomplished using the entire feature space instead of a subset.

**Comprehensive Dataset**: Multinomial Logistic regression classifier achieved the best test accuracy of **60.7%**, combined with One-Hot encoding for categorical variables, normalization for feature scaling, and random forest-based feature selection, which selected 110 out of the 2,105 initial features.  
*** No categorical encoding method was needed in the grid search due to the characteristics of the considered dataset


![mat](https://user-images.githubusercontent.com/80990030/228986342-7a2dc27f-e314-4951-8937-8b6dc32f1606.png)
Figure 9 represents the confusion matrix for the best model trained on the comprehensive dataset, which is a tabular summary of its performance. As emphasized in the literature, football match prediction presents unique challenges due to its low-scoring nature, especially when considering draws in a multiclass design. The main issue in multiclass football match prediction is the complexity added when considering draws, which is the class where the comprehensive model severely underperforms. 



![comp](https://user-images.githubusercontent.com/80990030/228985611-a1ba0764-7b5a-479f-944a-6e3816dbdab0.png)
Figure 10 offers valuable insight into the effectiveness of different types of football-related knowledge in predicting outcomes. Specifically, it enables us to compare and rank the predictive power associated with various feature sub-spaces considered in this study. The chart illustrates the best performances obtained using different feature sets. Every model trained on each of the sub-dataset significantly improved the test accuracy over the benchmark model's performance, demonstrating the effectiveness of this feature subset in predicting the outcome of football matches. The benchmark model solely predicts the majority class in the dataset, which is a home win, achieving an accuracy of 44.3%.
The plot also provides information on the number of features selected via dimensionality reduction methods and classification learning algorithm, which yielded the best results. 
Interestingly, the best performances across different feature sets were obtained predominantly by implementing either Multinomial Logistic regression classifiers or k-Nearest Neighbors classifiers, while only once with a Random Forest classifier and never with the other considered classification algorithms. 
Furthermore, it is worth noting that feature selection methods outperformed feature extraction techniques using Principal Component Analysis in all the analyzed feature sets. These findings suggest that Multinomial Logistic regression or k-Nearest Neighbors classifiers, combined with feature selection methods, are most effective in predicting football outcomes. In contrast, feature extraction techniques using PCA may be less effective in this context.  
The model trained on the comprehensive dataset outperformed all the models trained on sub-datasets. This model also performed better than most prior multiclass match prediction studies in the literature.  



<a id='conclusions'></a>
## Conclusions

In this study, we aimed to expand upon existing research on football match prediction by using a more extensive feature set and designing a strategy to compare the predictive value of different forms of football-related knowledge. Our results demonstrated that the model using the comprehensive feature  space achieved a test accuracy of 60.7%, outperforming many prior studies on multiclass match prediction. However, the comprehensive model's performance corroborated the earlier literature's reflections on the intrinsic complexity in predicting football match outcomes in a multiclass configuration due to complications added when considering draws which was the class where our model severely underperformed. We also partitioned the comprehensive dataset into seven smaller feature sets based on shared information-specific characteristics, using a standard set of preprocessing techniques and classification algorithms that allowed us to compare the predictive power of various types of knowledge. Notably, our findings showed that betting odds were the most valuable information type for the prediction task, supporting prior research.
Testing the comprehensive model's effectiveness in generating a profit through betting strategies represents a natural evolution of our analysis. Moreover, this research could provide a helpful roadmap for navigating the vastness of football-related information for future studies. 

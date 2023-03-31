# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For ML 
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


def plot_confusion_matrix(y_pred, y_test, color_map):
    """
    Plot a confusion matrix based on the predicted and actual labels.
    Args:
        y_pred (array-like): Predicted labels for the test set.
        y_test (array-like): True labels for the test set.
        color_map (str or matplotlib colormap): The colormap to use for the heatmap.
    """    
    # Compute and plot the Confusion Matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig = plt.subplots(figsize=(18, 7))
    plot = sns.heatmap(cm, annot=True, fmt='g',cmap=color_map, linewidth=1)  
    # Set the axes labels, title, and ticks
    plot.set_xlabel('Predicted labels')
    plot.set_ylabel('True labels')
    plot.set_title('Confusion Matrix: ')
    plot.xaxis.set_ticklabels(['Draw', 'Home Win', 'Away Win'])
    plot.yaxis.set_ticklabels(['Draw', 'Home Win', 'Away Win'])
    plt.show()


def select_features_rf(X_train, y_train, X_test, n):
    """
    Select the top n features using a random forest classifier.
    Args:
        X_train (array-like): Training data.
        y_train (array-like): Target labels for the training data.
        X_test (array-like): Test data.
        n (int): The number of top features to select.
    Returns:
        tuple: A tuple containing the transformed training and test data, and the fitted feature selector object.
    """
    # Create Random Forest-based feature selector object + fit on training set
    fs = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=14), max_features=n)
    fs.fit(X_train, y_train)
    # Transform and return training and test data using fitted feature selector
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def select_features_f(X_train, y_train, X_test, n):
    """
    Select the top n features using the f_classif function.
    Args:
        X_train (array-like): Training data.
        y_train (array-like): Target labels for the training data.
        X_test (array-like): Test data.
        n (int): The number of top features to select.
    Returns:
        tuple: A tuple containing the transformed training and test data, and the fitted feature selector object.
    """
    # Create feature selector using the f_classif function + fit on training data
    fs = SelectKBest(score_func=f_classif, k=n)
    fs.fit(X_train, y_train)
    # Transform and return training and test data using fitted feature selector
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def select_features_mutual(X_train, y_train, X_test, n):
    """
    Select the top n features using mutual information.
    Args:
        X_train (array-like): Training data.
        y_train (array-like): Target labels for the training data.
        X_test (array-like): Test data.
        n (int): The number of top features to select.
    Returns:
        tuple: A tuple containing the transformed training and test data, and the fitted feature selector object.
    """
    # Create a feature selector using mutual information + fit on training data
    fs = SelectKBest(score_func=mutual_info_classif, k=n)
    fs.fit(X_train, y_train)
    # Transform and return training and test data using fitted feature selector
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def grinding_store(estimator, grid_parameters, X_tr, y_tr, X_te, y_te):
    """
    Fits an estimator using GridSearchCV to optimize hyperparameters based on a parameter grid.
    Args:
        estimator (sklearn estimator): The estimator to be optimized using GridSearchCV.
        grid_parameters (dict): The parameter grid to search over.
        X_tr (array-like): Training data.
        y_tr (array-like): Training labels.
        X_te (array-like): Test data.
        y_te (array-like): Test labels.
    """
    gs = GridSearchCV(estimator=estimator, param_grid=grid_parameters, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=14), scoring='accuracy', refit=True, n_jobs=-1)
    gs.fit(X_tr,y_tr)
    # Make y_pred from the best model
    model_BEST = gs.best_estimator_
    model_BEST.fit(X_tr, y_tr)
    y_pred = model_BEST.predict(X_te)
    # Model's Accuracy 
    return {'best model': gs.best_estimator_, 'Train acc.': gs.best_score_, 'Test acc.': metrics.accuracy_score(y_te, y_pred)}


def grinding_print(estimator, grid_parameters, X_tr, y_tr, X_te, y_te):
    """
    Fits an estimator using GridSearchCV to optimize hyperparameters based on a parameter grid and prints the results.
    Args:
        estimator (sklearn estimator): The estimator to be optimized using GridSearchCV.
        grid_parameters (dict): The parameter grid to search over.
        X_tr (array-like): Training data.
        y_tr (array-like): Training labels.
        X_te (array-like): Test data.
        y_te (array-like): Test labels.
    """
    gs = GridSearchCV(estimator=estimator, param_grid=grid_parameters, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=14), scoring='accuracy', refit=True, n_jobs=-1)
    gs.fit(X_tr,y_tr)
    # Make y_pred from the best model
    model_BEST = gs.best_estimator_
    model_BEST.fit(X_tr, y_tr)
    y_pred = model_BEST.predict(X_te)
    # Model's Accuracy 
    print('Best Model: ', gs.best_estimator_, '\nBest Train Accuracy: ', gs.best_score_, 'Test Accuracy: ', metrics.accuracy_score(y_te, y_pred))



def models_complete_feat(X_training, y_training, X_testing, y_testing, estimators):
    """
    Iterates over a set of estimators and performs grid search with CV using the provided training and testing feature matrices and target vectors.
    Args:
        X_training (numpy.ndarray): Feature matrix for the training data.
        y_training (numpy.ndarray): Target variable vector for the training data.
        X_testing (numpy.ndarray): Feature matrix for the testing data.
        y_testing (numpy.ndarray): Target variable vector for the testing data.
        estimators (dictionary): Dictionary containing classification algorithms and respective hyperparameters 
    """
    for key, value in estimators.items():
        print('################', key, '################')
        grinding_print(estimator=value[0], grid_parameters=value[1], X_tr=X_training, y_tr=y_training, X_te=X_testing, y_te=y_testing)
    return None

def models_pca(X_training, y_training, X_testing, y_testing, estimators):
    """
    Perform PCA for feature extraction on the training and test set and evaluate performance of different ML models using the reduced feature space. 
    Args: 
        X_training (numpy.ndarray): Feature matrix for the training data.
        y_training (numpy.ndarray): Target variable vector for the training data.
        X_testing (numpy.ndarray): Feature matrix for the testing data.
        y_testing (numpy.ndarray): Target variable vector for the testing data.
        estimators (dictionary): Dictionary containing classification algorithms and respective hyperparameters
    """
    # Iterate through the dictionary of estimators
    for key, value in estimators.items():
        model_list = []
        # Iterate through multiple PCA component values
        for comp in [0.65, 0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99]:
            # Instantiate a PCA object with the given number of components + fit train/test sets
            pca = PCA(n_components=comp, svd_solver='full')
            X_tr_pca = pca.fit_transform(X_training)
            X_te_pca = pca.transform(X_testing)
            # Employ grid search to find the best performing model for each combination
            ist_dict = grinding_store(estimator=value[0], grid_parameters=value[1], X_tr=X_tr_pca, y_tr=y_training, X_te=X_te_pca, y_te=y_testing)
            ist_dict['features'] = comp
            model_list.append(ist_dict) 
        # Print the best performing model for each classifier and its test accuracy
        print(max(model_list, key=lambda x: x['Test acc.']))
    return None

def models_feature_selection(X_train, y_train, X_test, y_test, estimators, sel_range, feat_sel):
    """
    Fits a set of estimators with feature selection using a specified technique and a range of selected features. Returns the model with the highest test accuracy for each estimator.
    Args:
        X_training (numpy.ndarray): Feature matrix for the training data.
        y_training (numpy.ndarray): Target variable vector for the training data.
        X_testing (numpy.ndarray): Feature matrix for the testing data.
        y_testing (numpy.ndarray): Target variable vector for the testing data.
        estimators (dictionary): Dictionary containing classification algorithms and respective hyperparameters
        sel_range (list of int or int): Number of features to select using the indicated feature selection method
        feat_sel (str): The feature selection method to use. Can be 'rf', 'f', or 'mi'.
    """
    for key, value in estimators.items():
        model_list = []
        for n in sel_range:
            try:
                feature_selector = {
                'rf': select_features_rf,
                'f': select_features_f,
                'mi': select_features_mutual
                }[feat_sel]
            except KeyError:
                raise ValueError(f'Invalid feature selection method: {feat_sel}')

            X_train_fs, X_test_fs, fs = feature_selector(X_train, y_train, X_test, n)
            ist_dict = grinding_store(estimator=value[0], grid_parameters=value[1], X_tr=X_train_fs, y_tr=y_train, X_te=X_test_fs, y_te=y_test)
            ist_dict['features'] = n
            model_list.append(ist_dict) 
        print(max(model_list, key=lambda x: x['Test acc.']))
    return None
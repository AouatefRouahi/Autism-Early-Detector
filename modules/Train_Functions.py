import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix 
from sklearn.model_selection import GridSearchCV

import pickle

import os
import time

model_path = 'models/'
report_path = 'models/'
seed = 0

# ***********************************************************************************
def missing(dataset):
    missing = pd.DataFrame(columns=['Variable', 'n_missing', 'p_missing'])

    miss = dataset.isnull().sum() # series

    missing['Variable'] = miss.index
    missing['n_missing'] = miss.values
    missing['p_missing'] = round(100*miss/dataset.shape[0],2).values

    return missing.sort_values(by='n_missing')

#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************

def train_test_val(X, y, train_ratio, test_ratio, val_ratio):
    assert sum([train_ratio, test_ratio, val_ratio])==1.0, "wrong given ratio, all ratios have to sum to 1.0"
    assert X.shape[0]==len(y), "X and y shape mismatch"
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, 
                                                      test_size = 1-train_ratio, 
                                                      random_state=seed,
                                                      stratify =y)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, 
                                                      test_size=test_ratio/(test_ratio + val_ratio), 
                                                      random_state=seed,
                                                      stratify =y_tmp)
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_val(X, y, val_ratio):
    assert val_ratio<1 and val_ratio>0, "wrong given ratio"
    assert X.shape[0]==len(y), "X and y shape mismatch"
    
    X_train, X_val, y_train, y_val= train_test_split(X, y, 
                                                      test_size = val_ratio, 
                                                      random_state=seed,
                                                      stratify =y)
    return X_train, X_val, y_train, y_val

#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************

def intermediate_steps(clf:Pipeline, X_train, step=None):
    if step != None:
        print('////////////////////////////////////////////////////////////////////////////////////////')
        print('/////////////////////////////Intermediate Steps: TF-IDF Vectorizer//////////////////////')
        tfidf_vect = clf.named_steps[step]
        #tfidf_vect = clf.named_steps['preprocessor'].named_transformers_[step]
        print(f"Train_n_features : {len(tfidf_vect.vocabulary_)}")
        train_tfidf = tfidf_vect.fit_transform(X_train)
        X_train_tfidf = pd.DataFrame(train_tfidf.todense(),columns=tfidf_vect.get_feature_names()).T
        #display(X_train_tfidf)
        common_X_train_tfidf = X_train_tfidf.mean(axis=1).sort_values(ascending=False)
        print("most common features\n")
        print(common_X_train_tfidf[:30])
        print('////////////////////////////////////////////////////////////////////////////////////////')
    else:
        print("No intermediate step is supplied !!")
    
def check_params_exist(esitmator, params_keyword):
    all_params = esitmator.get_params().keys()
    available_params = [x for x in all_params if params_keyword in x]
    if len(available_params)==0:
        return "No matching params found!"
    else:
        return available_params
#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************

# inspired by: 
#https://medium.com/towards-data-science/binary-and-multiclass-text-classification-auto-detection-in-a-model-test-pipeline-938158854943
#https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/blob/master/01_tf-idf.ipynb

def classifier(clf,clf_name, X_train, y_train, X_val, y_val, stage, intermediate=False, step=None):
    results_path = report_path+stage+'/'+clf_name+'.txt'
    models_path = model_path+stage+'/'+clf_name+'.pickle'
    report = clf_name +'\n'
    
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_end = time.time() - train_start
    
    report = report + "Training time : %.3f s\n" %(train_end) 
    
    #*******************************************
    if intermediate:
        intermediate_steps(clf, X_train, step)
       
    #*******************************************

    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    
    report = report + "\n\t\tAccuracy\n"
    report = report + "Train Accuracy: {:.2f}\n".format(100 * accuracy_score(y_train, y_pred_train))
    report = report + "Validation Accuracy: {:.2f}\n".format(100 * accuracy_score(y_val, y_pred_val))

    report = report + "\n\t\tF1-Score\n"
    report = report + "Train F1-Score: {:.2f}\n".format(100 * f1_score(y_train, y_pred_train))
    report = report + "Validation F1-Score: {:.2f}\n".format(100 * f1_score(y_val, y_pred_val))

    report = report + "\nTrain: Classification Report\n" + classification_report(y_train, y_pred_train)
    report = report + "\nValidation: Classification Report\n" + classification_report(y_val, y_pred_val)

    print(f"\nfind the numerical results at this path: \n\t{results_path}\n")
    
    print("\nConfusion Matrix\n")
    plot_confusion_matrix(clf, X_val, y_val)  
    
    #save results 
    if os.path.exists(results_path):
        os.remove(results_path)

    textfile = open(results_path, "w")
    textfile.write(report)
    textfile.close()

    #save model
    if os.path.exists(models_path):
        os.remove(models_path)
    with open(models_path, 'wb') as f:
        pickle.dump(clf, f)
                   
    print(f"\nfind the model.pickle saved at this path: \n\t{models_path}\n")
    
    return results_path, models_path


def grid_search(clf, X_train, X_val,y_train, y_val, param_grid):
    '''
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    validation_indexes = [-1]*len(X_train) + [0]*len(X_val)
    ps = PredefinedSplit(test_fold=validation_indexes)
    '''
    X = pd.concat((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    validation_indexes = [-1]*len(X_train) + [0]*len(X_val)
    ps = PredefinedSplit(test_fold=validation_indexes)
    
    grid_search = GridSearchCV(
        clf, param_grid, cv=ps, 
        scoring='f1', return_train_score=True, 
        n_jobs=-1, verbose=5
    )
    print('\nFitting \n')
    grid_search.fit(X, y)
    
    print('\nBest Classifier \n') 
    best_clf = grid_search.best_estimator_
    print('Best Params \n') 
    print(grid_search.best_params_)
    print('Best score \n') 
    print(grid_search.best_score_)
 
    return best_clf

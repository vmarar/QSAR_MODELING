# +
import pandas as pd 
from sklearn.metrics  import f1_score,accuracy_score, precision_score, recall_score
pd.set_option('display.max_colwidth',50000)
import rdkit.ML.Descriptors.MoleculeDescriptors as Calc
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import itertools
import warnings
warnings.filterwarnings('ignore')
import pickle as pickle
import joblib
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold 

from descriptor_generation_essentials import generate_descriptors
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from tensorflow import keras
#from keras.models import Model
#from keras.layers import Input, add
#from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
#from keras import regularizers
#from keras.regularizers import l2
#from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import DistanceMetric
import os


# -

# # Find Ideal Threshold

# +
def assign_label(x,indicator,good_value):
    '''
    This function converts values into labels 
    based on indicator and good_value,
    
    good_value : threshold for 'GOOD' labels 
    indicator : indicator for good_value
    
    '''
    if indicator == 'less':
        if x < good_value :
            return 'GOOD'
        else:
            return 'BAD'
        
    if indicator == 'greater':
        if x > good_value :
            return 'GOOD'
        else:
            return 'BAD'
        

def find_threshold(data,target,starting_thresh, indicator, skip):
    '''
    This function takes a starting_threshold and finds a threshold near the 
    starting_threshold such that the proporion of each label is atleast 25% of the dataset. 
    
    data : dataset containing assay value 
    target : assay column name 
    starting_thresh : ideal threshold value for 'GOOD' classification 
    indicator : 'greater' than starting_thresh / 'less' than starting_thresh , for 'GOOD' classification
    skip : if 'skip', starting_thresh must be used and other values will not be tested
    
    '''
    # how many values to check above
    above = starting_thresh + 40
    
    # max lowest value you can check
    for i in range(40):
        a = starting_thresh - i
        if a > 0:
            below = a
            
    acc_len = len(data)*.25
    
    best_thresh = []
    
    if skip=='skip':
        data['TARGET_QUALITY'] = data[target].apply(lambda x:assign_label(x,indicator,starting_thresh))
        new_thresh = starting_thresh
        
    else:
    
        if indicator == 'greater':
            for i in range(starting_thresh,above):
                data['TARGET_QUALITY'] = data[target].apply(lambda x:assign_label(x,indicator,i))
                if len(data['TARGET_QUALITY'].value_counts().sort_values(ascending=False)) > 1:
                    check1 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[0]
                    check2 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[1]
                    if check1 >= acc_len and check2 >= acc_len:
                        best_thresh.append(i)


            if not best_thresh:
                for i in reversed(range(below,starting_thresh)):
                    data['TARGET_QUALITY'] = data[target].apply(lambda x:assign_label(x,indicator,i))
                    if len(data['TARGET_QUALITY'].value_counts().sort_values(ascending=False)) > 1:
                        check1 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[0]
                        check2 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[1]
                        if check1 >= acc_len and check2 >= acc_len:
                            best_thresh.append(i) 


        if indicator ==  'less':
            for i in reversed(range(below,starting_thresh)):
                data['TARGET_QUALITY'] = data[target].apply(lambda x:assign_label(x,indicator,i))
                if len(data['TARGET_QUALITY'].value_counts().sort_values(ascending=False)) > 1:
                    check1 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[0]
                    check2 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[1]
                    if check1 >= acc_len and check2 >= acc_len:
                        best_thresh.append(i)


            if not best_thresh:
                for i in range(starting_thresh,above):
                    data['TARGET_QUALITY'] = data[target].apply(lambda x:assign_label(x,indicator,i))
                    if len(data['TARGET_QUALITY'].value_counts().sort_values(ascending=False)) > 1:
                        check1 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[0]
                        check2 = data['TARGET_QUALITY'].value_counts().sort_values(ascending=False).iloc[1]
                        if check1 >= acc_len and check2 >= acc_len:
                            best_thresh.append(i)


        if len(best_thresh) < 1:
            best_thresh.append(starting_thresh)

        #new_thresh = min(best_thresh)
        diffs = []
        for i in best_thresh:
            diff = abs(starting_thresh - i)
            diffs.append(diff)

        indx = diffs.index(min(diffs))
        new_thresh = best_thresh[indx]

        data['TARGET_QUALITY']= data[target].apply(lambda x:assign_label(x,indicator,new_thresh))
        print(data['TARGET_QUALITY'].value_counts().sort_values(ascending=False))

    return data, new_thresh, indicator
# -

# # Variable Selection

# +
# Ref: http://easymachinelearning.blogspot.in/p/sparse-auto-encoders.html

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin

def _binary_KL_divergence(p, p_hat):
    return (p * np.log(p / p_hat)) + ((1 - p) * np.log((1 - p) / (1 - p_hat)))

def _logistic(X):
    return 1. / (1. + np.exp(np.clip(-X, -30, 30)))
    
def _d_logistic(X):
    return X * (1 - X)
    
class Autoencoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_hidden=25,
        learning_rate=0.3, alpha=3e-3, beta=3, sparsity_param=0.1,
        max_iter=20, verbose=False, random_state=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta
        self.sparsity_param = sparsity_param
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def _init_fit(self, n_features):
        """
        Weights' initilization
        """
        rng = check_random_state(self.random_state)
        self.coef_hidden_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, n_features))
        self.intercept_hidden_ = rng.uniform(-1, 1, self.n_hidden)
        self.intercept_output_ = rng.uniform(-1, 1, n_features)

    def _unpack(self, theta, n_features):
        N = self.n_hidden * n_features
        self.coef_hidden_ = np.reshape(theta[:N],
                                      (n_features, self.n_hidden))
        self.coef_output_ = np.reshape(theta[N:2 * N],
                                      (self.n_hidden, n_features))
        self.intercept_hidden_ = theta[2 * N:2 * N + self.n_hidden]
        self.intercept_output_ = theta[2 * N + self.n_hidden:]

    def _pack(self, W1, W2, b1, b2):
        return np.hstack((W1.ravel(), W2.ravel(),
                          b1.ravel(), b2.ravel()))

    def transform(self, X):
        return _logistic(np.dot(X, self.coef_hidden_) + self.intercept_hidden_)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        self._init_fit(n_features)
        self._backprop_lbfgs(
                X, n_features, n_samples)
        return self

    def backprop(self, X, n_features, n_samples):
        # Forward pass
        a_hidden = _logistic(np.dot(X, self.coef_hidden_)
                                      + self.intercept_hidden_)
        a_output = _logistic(np.dot(a_hidden, self.coef_output_)
                                      + self.intercept_output_)
        # Compute average activation of hidden neurons
        p = self.sparsity_param
        p_hat = np.mean(a_hidden, 0)
        p_delta  = self.beta * ((1 - p) / (1 - p_hat) - p / p_hat)
        # Compute cost 
        diff = X - a_output
        cost = np.sum(diff ** 2) / (2 * n_samples)
        # Add regularization term to cost 
        cost += (0.5 * self.alpha) * (
            np.sum(self.coef_hidden_ ** 2) + np.sum(
                self.coef_output_ ** 2))
        # Add sparsity term to the cost
        cost += self.beta * np.sum(
            _binary_KL_divergence(
                p, p_hat))
        # Compute the error terms (delta)
        delta_output = -diff * _d_logistic(a_output)
        delta_hidden = (
            (np.dot(delta_output, self.coef_output_.T) +
             p_delta)) * _d_logistic(a_hidden)
        #Get gradients
        W1grad = np.dot(X.T, delta_hidden) / n_samples 
        W2grad = np.dot(a_hidden.T, delta_output) / n_samples
        b1grad = np.mean(delta_hidden, 0) 
        b2grad = np.mean(delta_output, 0) 
        # Add regularization term to weight gradients 
        W1grad += self.alpha * self.coef_hidden_
        W2grad += self.alpha * self.coef_output_
        return cost, W1grad, W2grad, b1grad, b2grad

    def _backprop_lbfgs(self, X, n_features, n_samples):
        #Pack the initial weights 
        #into a vector
        initial_theta = self._pack(
            self.coef_hidden_,
            self.coef_output_,
            self.intercept_hidden_,
            self.intercept_output_)
        #Optimize the weights using l-bfgs
        optTheta, _, _ = fmin_l_bfgs_b(
            func=self._cost_grad,
            x0=initial_theta,
            maxfun=self.max_iter,
            disp=self.verbose,
            args=(X,
                n_features,
                n_samples))
        #Unpack the weights into their
        #relevant variables
        self._unpack(optTheta, n_features)

    def _cost_grad(self, theta, X, n_features,
                   n_samples):
        self._unpack(theta, n_features)
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(
            X, n_features, n_samples)
        return cost, self._pack(W1grad, W2grad, b1grad, b2grad)


# +
# Remove Low Variance Variables
def variance_selection(x , y, xtest, features):
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit(x, y)
        x_train = sel.transform(x)
        x_test = sel.transform(xtest)
        
        cols = sel.get_support(indices=True)
        features_df_new = features.iloc[:,cols]

        return x_train, x_test, features_df_new

# Autoencoder Based Selection 
def Vanilla_Autoencoder(x_test, x_train, n_hidden):
        ae = Autoencoder(max_iter=200,sparsity_param=0.1,
                                        beta=3,n_hidden = n_hidden,alpha=3e-3,
                                        verbose=True, random_state=1)
        # Train and extract features
        extracted_features = ae.fit(x_train)
        x_test = ae.transform(x_test)
        x_train = ae.transform(x_train)

        return x_test, x_train, ae

# Recursive Feature Elimination 
def RFE_func(X_train, y_train, X_test, y_test, features):
        point = 0 
        
        if X_train.shape[1] < 20:
            length = X_train.shape[1]
        else:
            length=20
            
        print(len(features.columns.tolist()))
        for i in reversed(range(1,length)):
            model = RandomForestClassifier()
            rfe = RFE(estimator=model, n_features_to_select=i)
            fit = rfe.fit(X_train, y_train)

            y_hat = rfe.predict(X_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            if point < score:

                point = score
                X_train_scaled = fit.transform(X_train)
                X_test_scaled = fit.transform(X_test)

                cols = fit.get_support(indices=True)
                features_final = features.iloc[:,cols]
                
        return X_train_scaled, X_test_scaled, features_final 
    
# Scale and segment data
def Split_and_Scale(data, features, target, scale):
    X = data[features[0]]
    y = data[target]
    
    if scale == 'yes':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else: 
        scaler = None
    
    return X, y, scaler


# -

# # ML Modeling And Optimization

# +
from sklearn.metrics import precision_score, make_scorer
custom_scorer = make_scorer(precision_score, greater_is_better=True,  pos_label='GOOD')

def Model_Selection(name_list, x_test, x_train, y_test, y_train, cv, oversampling):
    
    if oversampling == 'True':
        sm = SMOTE(random_state = 2)
        x_train, y_train = sm.fit_resample(x_train, y_train.ravel())
    
    model_scores = []
    model_and_params = []
    
    for i in name_list:
        if i =='Random_Forest':
        # Random Forest Classifier
            balanced_data ='no'
            n_estimators = [100, 250, 500, 750, 1000]
            max_features = ['auto', 'sqrt']
            criterion = ['gini', 'entropy']

            if balanced_data == 'yes':
                class_weight = [None]
            else:
                class_weight = [None,'balanced',
                                {0:.9, 1:.1}, {0:.8, 1:.2}, {0:.7, 1:.3}, {0:.6, 1:.4},
                                {0:.4, 1:.6}, {0:.3, 1:.7}, {0:.2, 1:.8}, {0:.1, 1:.9}]
            random_state = [24]
            param_grid = {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'criterion': criterion,
                          'random_state': random_state,
                          'class_weight': class_weight}
            try: 
                rf = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
                rf.fit(x_train, y_train)
                y_hat = rf.predict(x_test)
                score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
                model_scores.append(['Random_Forest', score])
                model_and_params.append([RandomForestClassifier(), rf.best_params_])
            except:
                try: 
                    rf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
                    rf.fit(x_train, y_train)
                    y_hat = rf.predict(x_test)
                    score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
                    model_scores.append(['Random_Forest', score])
                    model_and_params.append([RandomForestClassifier(), rf.best_params_])
                except: 
                    model_scores.append(['Random_Forest_SKIPPED', 0])
                    model_and_params.append([RandomForestClassifier(), None])
                    

        if i == 'Logistic_Regression':
        # Logistic Regression
            param_grid = {'penalty' : ['l1', 'l2'],
                'C' : np.logspace(-4, 4, 20),
                'solver' : ['liblinear']}

            lr = GridSearchCV(LogisticRegression(), param_grid, n_jobs=1, scoring=custom_scorer, cv=cv, verbose=1)
            lr.fit(x_train, y_train)
            y_hat = lr.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Logistic_Regression', score])
            model_and_params.append([LogisticRegression(), lr.best_params_])

        if i == 'Naive_Bayes':
            # Naive Bayes
            nb = GaussianNB()
            score = cross_val_score(nb, x_train, y_train, cv=cv, scoring=custom_scorer)
            nb.fit(x_train, y_train) 
            y_hat = nb.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Naive_Bayes', score])
            model_and_params.append([GaussianNB(), None])

        if i == 'SVM':
            # Support Vector Machine
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']
                           }
            svm = GridSearchCV(SVC(), param_grid, n_jobs=1, scoring=custom_scorer, cv=cv, verbose=1)
            svm.fit(x_train, y_train)
            y_hat = svm.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['SVM', score])
            model_and_params.append([SVC(), svm.best_params_])

        if i == 'KNN':
            # K-Nearest-Neighbors                                     
            param_grid = {
                'n_neighbors' : [i for i in range(1,50,5)],
                'weights':['uniform','distance'],
                'metric':['euclidean','manhattan']
            }
            knn = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=1, scoring=custom_scorer, cv=cv, verbose=1)
            knn.fit(x_train, y_train)
            y_hat = knn.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['KNN', score])
            model_and_params.append([KNeighborsClassifier(), knn.best_params_])

        if i == 'Decision_Tree_Classifier':
            # Decision Tree Classifier
            param_grid = { 
                'criterion': ['gini', 'entropy'],
                'max_depth' : [i for i in range(5,100,10)],
                'min_samples_split':[i for i in range(2,10)],
                'min_samples_leaf' :[i for i in range(2,10)]
            }
            dt = GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=1, scoring=custom_scorer, cv=cv, verbose=1)
            dt.fit(x_train, y_train) 
            y_hat = dt.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Decision_Tree_Classifier', score])
            model_and_params.append([DecisionTreeClassifier(), dt.best_params_])


    model_scores = pd.DataFrame(model_scores)
    model_scores_index = model_scores[1].tolist()
    names_index = model_scores[0].tolist()
    max_value = max(model_scores_index)
    max_index = model_scores_index.index(max_value) 

    #print(names_index[max_index], model_scores_index[max_index])
    return model_and_params[max_index],[names_index[max_index], model_scores_index[max_index]], model_scores, model_and_params

# -
def Model_Selection_Boosted(name_list, x_test, x_train, y_test, y_train, cv, oversampling):
    
    if oversampling == 'True':
        sm = SMOTE(random_state = 2)
        x_train, y_train = sm.fit_resample(x_train, y_train.ravel())
    
    model_scores = []
    model_and_params = []

    
    for i in name_list:
        print(i)
        if i == 'Random_Forest': 
        # Random Forest With Boosting Algorithim 
            balanced_data ='no'
            n_estimators = [100, 250, 500, 750, 1000]
            max_features = ['auto', 'sqrt']
            criterion = ['gini', 'entropy']

            if balanced_data == 'yes':
                class_weight = [None]
            else:
                class_weight = [None,'balanced',
                                {0:.9, 1:.1}, {0:.8, 1:.2}, {0:.7, 1:.3}, {0:.6, 1:.4},
                                {0:.4, 1:.6}, {0:.3, 1:.7}, {0:.2, 1:.8}, {0:.1, 1:.9}]
            random_state = [24]

            abc = AdaBoostClassifier(base_estimator=RandomForestClassifier())
            param_grid = {'base_estimator__n_estimators': n_estimators,
                          'base_estimator__max_features': max_features,
                          'base_estimator__criterion': criterion,
                          'base_estimator__random_state': random_state,
                          'base_estimator__class_weight': class_weight}

            rfabc = GridSearchCV(abc, param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
            rfabc.fit(x_train,y_train)
            y_hat = rfabc.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Random_Forest_Boosted', score])
            model_and_params.append([RandomForestClassifier(), rfabc.best_params_])

        if i == 'Logistic_Regression':
        # Logistic Regression With Boosting Algorithim
            abc = AdaBoostClassifier(base_estimator=LogisticRegression())
            param_grid = {'base_estimator__penalty' : ['l1', 'l2'],
                'base_estimator__C' : np.logspace(-4, 4, 20),
                'base_estimator__solver' : ['liblinear']}
            
            lrabc = GridSearchCV(abc, param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
            lrabc.fit(x_train,y_train)
            y_hat = lrabc.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Logistic_Regression_Boosted', score])
            model_and_params.append([LogisticRegression(), lrabc.best_params_])

        if i == 'SVM':
        # Support Vector Machine With Boosting Algorithim
            abc = AdaBoostClassifier(base_estimator=SVC())
            param_grid = {'base_estimator__C': [0.1, 1, 10, 100, 1000],
              'base_estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'base_estimator__kernel': ['rbf']}

            svmabc = GridSearchCV(abc, param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
            svmabc.fit(x_train,y_train)
            y_hat = svmabc.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Stochastic_Gradient_Descent_Boosted', score])
            model_and_params.append([SVC(), svmabc.best_params_])

        if i == 'KNN':
        # K-Nearest-Neighbors  
            abc = AdaBoostClassifier(base_estimator=KNeighborsClassifier())

            param_grid = {
                'base_estimator__n_neighbors' : [i for i in range(1,50,5)],
                'base_estimator__weights':['uniform','distance'],
                'base_estimator__metric':[DistanceMetric.get_metric('euclidean'),DistanceMetric.get_metric('manhattan')]
            }
            knndabc = GridSearchCV(abc, param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
            knndabc.fit(x_train,y_train)
            y_hat = knndabc.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['KNN_Boosted', score])
            model_and_params.append([KNeighborsClassifier(), knnabc.best_params_])

        if i =='Decision_Tree_Classifier':
        # Decision Tree Classifier
            abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

            param_grid = { 
                'base_estimator__criterion': ['gini', 'entropy'],
                'base_estimator__max_depth' : [i for i in range(5,100,10)],
                'base_estimator__min_samples_split':[i for i in range(2,10)],
                'base_estimator__min_samples_leaf' :[i for i in range(2,10)]
            }
            dtdabc = GridSearchCV(abc, param_grid, n_jobs=1, scoring=custom_scorer,cv=cv, verbose=1)
            dtdabc.fit(x_train,y_train)
            y_hat = dtdabc.predict(x_test)
            score = precision_score(y_test,y_hat,average="binary", pos_label='GOOD',zero_division=1)
            model_scores.append(['Decision_Tree_Classifier_Boosted', score])
            model_and_params.append([DecisionTreeClassifier(), dtabc.best_params_])


    model_scores = pd.DataFrame(model_scores)
    model_scores_index = model_scores[1].tolist()
    names_index = model_scores[0].tolist()
    max_value = max(model_scores_index)
    max_index = model_scores_index.index(max_value) 

    #print(names_index[max_index], model_scores_index[max_index])
    return model_and_params[max_index], [names_index[max_index], model_scores_index[max_index]], model_scores, model_and_params


def prepare_data(data, i, scale):
    
    X, y, scaler = Split_and_Scale(data, [i], 'TARGET_QUALITY', scale)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    x_train, x_test, features = variance_selection(x_train , y_train, x_test, data[i])
    x_train, x_test, features = RFE_func(x_train, y_train, x_test, y_test, features)

    return x_train, x_test, y_train, y_test, features, scaler 


def modeling(i, data, scale):
    x_train, x_test, y_train, y_test, features, scaler = prepare_data(data, i, scale)
    
    #x_testae, x_trainae, ae = Vanilla_Autoencoder(x_test, x_train)
    #new_data_test = ae.transform(new_data[i[0]])
    
    # MODELS 
    a,b,c,d = Model_Selection(['Random_Forest'], x_test, x_train, y_test, y_train, cv=5, oversampling='False')
    model_best = a[0].set_params(**a[1],n_jobs=1)
    model_best.fit(x_train, y_train)
    #vals = model_best.predict(new_data[features.columns.tolist()])
    vals = model_best.predict(x_test)
    print('DONE')

    return [model_best, features.columns.tolist(), precision_score(vals, y_test, average='binary', pos_label='GOOD'), scaler, a[1]]


# # DEEP LEARNING APPROACH

# +
from tensorflow.keras import models
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

# np.reshape(np.asarray(y_train), (302, 1))
def r2(x, y):
    print(x,y)

    x = K.batch_flatten(x)
    y = K.batch_flatten(y)

    mean_x = K.mean(x)
    mean_y = K.mean(y)

    num = K.sum((x - mean_x) * (y - mean_y))
    num *= num

    denom = K.sum((x - mean_x) * (x - mean_x)) * \
        K.sum((y - mean_y) * (y - mean_y))

    return num / denom

def generate_model(input_shape):
    model = models.Sequential()

    model.add(Dense(4000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(2000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.10))
    model.add(Dense(2, activation='softmax', use_bias=True,
                    kernel_regularizer=l2(0.0001)))
    optimizer = SGD(lr=0.001, momentum=0.9, clipnorm=1.0)
    model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['val_loss'])
    return model

# +
# using autoencoder
#i = descs[1]
#X, y, scaler = Split_and_Scale(data, i, 'TARGET_QUALITY')
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#nn2 = generate_model(np.asarray(X_train)[0].shape)
#x_testae, x_trainae, ae = Vanilla_Autoencoder(x_test, x_train, n_hidden=X_train.shape[0]//5)
#new_data_test = ae.transform(new_data[i[0]])
#nn2.fit(np.asarray(X_train), np.asarray(y_train), epochs=12, batch_size=5)

from collections import defaultdict
import json
import sklearn
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from spacy import explain

import xgboost as xgb

import lime
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import shap
from anchor import anchor_tabular

import pickle
import os
import time
import copy

import warnings
warnings.filterwarnings("ignore")
#class Dataset(): #abstract class

# class RegressionDataset(Dataset):


class Classification:
    def __init__(self,dataset_name,X,Y,class_names):
        """
        dataset_name : unique name to the dataset
        X : columns containing data
        Y : column to be predicted
        class_names : name of classes/labels on the dataset
        feature_names : list of columns in the dataset used to train models
        categorical features : list of features that are not continous or categorical
        """
        self.dataset_name = dataset_name
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X, Y,test_size=0.3, random_state=0)
        self.class_names = class_names
        self.models = {"random_forest" : RandomForestClassifier(), 
                        "knn" : KNeighborsClassifier() , 
                        "MLP" : MLPClassifier(hidden_layer_sizes=(120, 120,120), activation='relu', solver='lbfgs', max_iter=10000,alpha=1e-5,tol=1e-5), 
                        "decision_tree" : DecisionTreeClassifier(), 
                        "xgboost" : xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)
                        }


    

            
            
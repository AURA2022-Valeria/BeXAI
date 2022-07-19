from cmath import nan
from collections import defaultdict
import json
import sklearn
import pandas as pd
import numpy as np
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
import shap
from anchor import anchor_tabular

import pickle
import os
import time

from classification import Classification

class TabularClassification(Classification):
    #TODO : divide the class to regreesion and classification datasets 
    #TODO : the test_size can be passed as a parameter. should it?
    
    def __init__(self,dataset_name,X,Y,feature_names,class_names,categorical_features=[]):
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_features_indexes = [X.columns.get_loc(feature) for feature in categorical_features] #array of indexes of categorical features
        
        le = sklearn.preprocessing.LabelEncoder()
        self.categorical_names = {} #dictionary -> (x : int, categories : array) where categories is an array of string representation of the different categories for features x

        for i,feature in enumerate(self.categorical_features):
            X[feature] = pd.Series(X[feature], dtype="string")

        for i,feature in enumerate(self.categorical_features):
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(X.iloc[:,X.columns.get_loc(feature)])
            self.categorical_names[self.categorical_features_indexes[i]] = list(le.classes_)
            X[feature] = list(map(int,le.transform(X[feature])))
        
        
        self.encoder = ColumnTransformer(
            [('OHE', OneHotEncoder(),categorical_features)],
            remainder = 'passthrough'
            ) 
        self.encoder.fit(X)

        #X is to be splitted after the label encoding steps 
        super().__init__(dataset_name,X,Y,class_names)

        #number of records to explain when measuring runtime
        self.n = min(1,self.X_test.shape[0])


        #pipelines using the encoder and ml_algorithms 
        self.pipelines = {}

        #setting up explainers
        self.lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        
        self.anchor_explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.X_train.values,
            self.categorical_names)

        self._load_models()
    
    def _load_models(self):
        """
        Loads trained models if they are saved otherwise trains new models
        """
        file_path = f'Models/{self.dataset_name}'

        for label,model in self.models.items():
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            model_file_path = f'{file_path}/{label}' 
            try:
                self.models[label] = pickle.load(open(model_file_path, "rb")) 
                self.pipelines[label] = make_pipeline(self.encoder,self.models[label])
            except:
                self.pipelines[label] = make_pipeline(self.encoder,self.models[label])
                self.pipelines[label].fit(self.X_train,self.Y_train)
                pickle.dump(self.models[label], open(model_file_path, "wb")) #save the trained model inorder not to train it again

        # for label,model in self.models.items():
        #     try:
        #         print(model.score(self.X_test,self.Y_test))
        #     except Exception as e:
        #         print(e)

    
    def _get_pipeline_predicfn(self,pipeline,probability=False):
        def format_and_predict(x):
            if len(x.shape) == 1:
                pdf = pd.DataFrame([x],columns=self.feature_names)
            else:
                pdf = pd.DataFrame(x,columns=self.feature_names)
            if not probability:
                return pipeline.predict(pdf)
            return pipeline.predict_proba(pdf)
        return format_and_predict

    def get_lime_explanation(self,index_to_explain,pipeline_predict_fn):
        row_to_explain = self.X_test.iloc[index_to_explain].values
        exp = self.lime_explainer.explain_instance(row_to_explain,pipeline_predict_fn,num_features=len(self.feature_names),top_labels=1)

        exp = exp.as_map()
        prediction = list(exp.keys())[0]
        exp_list = exp[prediction]
        return exp_list
    
    def get_shap_explanation(self,index_to_explain,shap_explainer,):
        row_to_explain = self.X_test.iloc[index_to_explain]
        shap_vals = shap_explainer.shap_values(row_to_explain)
        return shap_vals
    
    def get_anchor_explanation(self,index_to_explain,pipeline):
        row_to_explain = self.X_test.iloc[index_to_explain].values
        pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=False)
        exp = self.anchor_explainer.explain_instance(row_to_explain, pipeline_predict_fn, threshold=0.9)
        return exp.names()

        
    def get_explanations(self,index_to_explain = 10,output=False):
        """
        Generates explanation for an instance in the test data
        """
        #TODO: modularize the time measurement 
        #TODO: Should graph representation be included in time measurements?

        row_to_explain = self.X_test.iloc[index_to_explain]
        
        #lime explanation
        lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        
        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=True)
            exp = lime_explainer.explain_instance(row_to_explain.values, pipeline_predict_fn,num_features=min(6,len(self.feature_names)),top_labels=1)
            # lim_feature_weight = exp.as_map() #weigths given by lime for features
            
            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                exp.save_to_file(f'{path}/lime_{label}.html')
    

        #shap explanation
        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=True)
            pred_class = np.argmax(pipeline_predict_fn(row_to_explain), axis=1)[0]
            k_explainer = shap.KernelExplainer(pipeline_predict_fn, self.X_train)
            shap_values = k_explainer.shap_values(row_to_explain)
            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                shap.initjs()
                figure = shap.force_plot(k_explainer.expected_value[pred_class], shap_values[pred_class], row_to_explain,matplotlib = True, show = False)
                figure.savefig(f'{path}/shap_{label}.png')

        #anchor explanation
        explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.X_train.values,
            self.categorical_names)

        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=False)
            exp = explainer.explain_instance(row_to_explain.values, pipeline_predict_fn, threshold=0.8)

            if output == True:
                try:
                    exp.save_to_file(f'{path}/anchor_{label}.html')
                except:
                    print("Couldn't generate html for anchor")
                    print(f"{label} Anchor explanation")
                    print('Anchor: %s' % (' AND '.join(exp.names())))
                    print('Precision: %.2f' % exp.precision())
                    print('Coverage: %.2f' % exp.coverage())


    def get_average_runtime_explanation_lime(self):
        lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        average_time_lime = defaultdict(int)
        lime_weights = defaultdict(int)

        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=True)
            start = time.perf_counter()
            
            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]

                exp = lime_explainer.explain_instance(row_to_explain.values, pipeline_predict_fn,num_features=len(self.feature_names))
                feature_weight = exp.as_map() #weigths given by lime for features
                lime_weights[label] = feature_weight

            end = time.perf_counter()
            average_time_lime[label] += end - start
        
        for label in self.pipelines:
            average_time_lime[label] /= self.n
        return average_time_lime
    
    def get_average_runtime_explanation_shap(self):
        average_time_shap = defaultdict(int)
        shap_values = defaultdict(int)

        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=True)
            k_explainer = shap.KernelExplainer(pipeline_predict_fn, self.X_train)
            start = time.perf_counter()
            
            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]
                shap_values[label] = label_shap_values = k_explainer.shap_values(row_to_explain)[1]
            
            end = time.perf_counter()
            average_time_shap[label] = end - start
        
        for label in self.pipelines:
            average_time_shap[label] /= self.n
        return average_time_shap
    
    def get_average_runtime_explanation_anchor(self):
        explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.X_train.values,
            self.categorical_names)

        average_time_anchor = defaultdict(int)
        for label,pipeline in self.pipelines.items():
            start = time.perf_counter()

            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]
                pipeline_predict_fn = self._get_pipeline_predicfn(pipeline,probability=False)
                exp = explainer.explain_instance(row_to_explain.values, pipeline_predict_fn, threshold=0.85)

            end = time.perf_counter()
            average_time_anchor[label] = end - start
        
        for label in self.pipelines:
            average_time_anchor[label] /= self.n

        return average_time_anchor 

    def get_average_explanation_times(self):
        average_times = {
            "lime" : self.get_average_runtime_explanation_lime(),
            "shap" : self.get_average_runtime_explanation_shap(),
            "anchor" : self.get_average_runtime_explanation_anchor(),
        }

        average_path = "run_time"
        if not os.path.exists(average_path):
            os.makedirs(average_path)

        with open(f'{average_path}/{self.dataset_name}.json', "w") as outfile:
            json.dump(average_times, outfile)
        return average_times
    
    def calculate_explainers_fidelity(self):
        """
        calculates fidelity scores of lime, shap and anchor over all models
        """
        lime_fidelity = defaultdict(int)
        shap_fidelity = defaultdict(int)
        anchor_fidelity = defaultdict(int)


        #mean values to simulate adding a feature
        base = np.mean(self.X_train,axis=0)
        
        #use modes instead of mean for categorical data to simulate removing
        for categorical in self.categorical_features:
            categorical_index = self.feature_names.index(categorical)
            unique_vals,counts = np.unique(self.X_train[categorical], return_counts=True)
            mode_index = np.argmax(counts)
            base[categorical_index] = mode_index

        N = 5 #number of rows to calculcate fidelity on
        for label,pipeline in self.pipelines.items():
            pipeline_probability_predict_fn = self._get_pipeline_predicfn(pipeline,probability=True)
            k_explainer = shap.KernelExplainer(pipeline_probability_predict_fn, self.X_train)

            for index in range(N):
                x = self.X_test.iloc[index]
                pred_class = np.argmax(pipeline_probability_predict_fn(x), axis=1)[0]
                               
                #lime fidelity
                
                lime_explanation = self.get_lime_explanation(index,pipeline_probability_predict_fn)
                #find indexs of coefficients in decreasing order of value
                exp_list = sorted(lime_explanation, key=lambda x: x[0])
                exp_weight = [x[1] for x in exp_list]
                label_lime_weights =  np.array(exp_weight)

                ar = np.argsort(-label_lime_weights)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
                positive_weight_index = []
                pred_probs = np.zeros(x.shape[0])
                monotonicity_probs = []

                for ind in ar:
                    if label_lime_weights[ind] >= 0:
                        positive_weight_index.append(ind)
                    x_copy = x.copy(deep=True)
                    x_copy.iloc[ind] = base.iloc[ind]
                    x_copy_pr = pipeline_probability_predict_fn(x_copy)
                    pred_probs[ind] = x_copy_pr[0][pred_class]
                    monotonicity_probs.append(pred_probs[ind])
                    
                C = -np.corrcoef(label_lime_weights, pred_probs)[0,1] 
                if not np.isnan(C):
                    lime_fidelity[label] += C
                

                #shap fidelity
                label_shap_vals = self.get_shap_explanation(index,k_explainer)[pred_class]
                ar = np.argsort(-label_shap_vals)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
                pred_probs = np.zeros(x.shape[0])

                for ind in ar:
                    x_copy = x.copy(deep=True)
                    x_copy.iloc[ind] = base.iloc[ind]
                    x_copy_pr = pipeline_probability_predict_fn(x_copy)
                    pred_probs[ind] = x_copy_pr[0][pred_class]

                C = -np.corrcoef(label_shap_vals, pred_probs)[0,1] 
                if not np.isnan(C):
                    shap_fidelity[label] += C

                #anchor fidelity
                anchor_explanation = " ".join(list(self.get_anchor_explanation(index,pipeline)))
                anchor_indexes = [] #indexes of columns returned from anchor explantion

                for row in self.X_train.columns:
                    if row in anchor_explanation:
                        anchor_indexes.append(self.X_train.columns.get_loc(row))
                
                base_copy = base.copy(deep=True)
                base_copy_prediction = []

                for anchor_index in anchor_indexes:
                    base_copy.iloc[anchor_index] = x.iloc[anchor_index]
                    base_copy_prediction.append(pipeline_probability_predict_fn(base_copy)[0][pred_class])

                C = np.corrcoef(list(range(len(base_copy_prediction))), base_copy_prediction)[0,1] #as more anchor features are added prediction probability should increase
                if not np.isnan(C):
                    anchor_fidelity[label] += C
           

            lime_fidelity[label] /= N
            shap_fidelity[label] /= N
            anchor_fidelity[label] /= N 

        fidelity = {
            "lime" : lime_fidelity,
            "shap" : shap_fidelity,
            "anchor" : anchor_fidelity,
        }

        fidelity_path = "fidelity"
        if not os.path.exists(fidelity_path):
            os.makedirs(fidelity_path)

        with open(f'{fidelity_path}/{self.dataset_name}.json', "w") as outfile:
            json.dump(fidelity, outfile)

        return fidelity
    

        
        



    

        



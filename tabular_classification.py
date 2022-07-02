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

        #number of records to explain when measureing time
        self.n = min(1,self.X_test.shape[0])

        #data sampled from training dataset to measure faithfulness and average explaining time
        sample_data = np.random.choice(range(self.X_train.shape[0]))

        #pipeline using the encoder and ml_algorithms 
        self.pipelines = {}

        #setting up explainers
        self.lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)

        self.load_models()
    
    def load_models(self):
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

    
    def get_pipeline_predicfn(self,pipeline,probability=False):
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
        exp = self.lime_explainer.explain_instance(row_to_explain,pipeline_predict_fn,num_features=len(self.feature_names))
        # exp.save_to_file("lime.html")
        

        exp_list = exp.as_map()[1]
        exp_list = sorted(exp_list, key=lambda x: x[0])
        exp_weight = [x[1] for x in exp_list]
        return np.array(exp_weight)
        
    def get_explanations(self,index_to_explain = 10,output=False):
        """
        Explains blackbox models and measure the time taken for the explainers
        """
        #TODO: modularize the time measurement 
        #TODO: Should graph representation be included in time measurements?

        timer = {
            "lime" : {},
            "shap" : {},
            "anchor" : {},
        }

        row_to_explain = self.X_test.iloc[index_to_explain]
        
        #lime explanation
        lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        
        for label,pipeline in self.pipelines.items():
            start = time.perf_counter()
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            exp = lime_explainer.explain_instance(row_to_explain.values, pipeline_predict_fn,num_features=min(6,len(self.feature_names)),top_labels=1)
            feature_weight = exp.as_map() #weigths given by lime for features
            end = time.perf_counter()
            timer["lime"][f'{self.dataset_name} - {label}'] = end - start
            
            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                exp.save_to_file(f'{path}/lime_{label}.html')
    

        #shap explanation
        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            start = time.perf_counter()
            k_explainer = shap.KernelExplainer(pipeline_predict_fn, self.X_train)
            shap_values = k_explainer.shap_values(row_to_explain)
            end = time.perf_counter()
            timer["shap"][f'{self.dataset_name} - {label}'] = end - start

            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                shap.initjs()
                figure = shap.force_plot(k_explainer.expected_value[1], shap_values[1], row_to_explain,matplotlib = True, show = False)
                figure.savefig(f'{path}/shap_{label}.png')

        #anchor explanation
        explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.X_train.values,
            self.categorical_names)

        for label,pipeline in self.pipelines.items():
            start = time.perf_counter()
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=False)
            exp = explainer.explain_instance(row_to_explain.values, pipeline_predict_fn, threshold=0.8)
            end = time.perf_counter()
            timer["anchor"][f'{self.dataset_name} - {label}'] = end - start

            if output == True:
                print(f"{label} Anchor explanation")
                exp.save_to_file(f'{path}/anchor_{label}.html')
                print('Anchor: %s' % (' AND '.join(exp.names())))
                print('Precision: %.2f' % exp.precision())
                print('Coverage: %.2f' % exp.coverage())

        if output == True:
            #save the time it took to explain in a json file
            with open(f"Running_Time/explanation_time_{self.dataset_name}.json", "w") as outfile:
                json.dump(timer, outfile)
        return timer

    def get_lime_explanations(self):
        lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        average_time_lime = defaultdict(int)
        lime_weights = defaultdict(int)

        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]
                start = time.perf_counter()

                exp = lime_explainer.explain_instance(row_to_explain.values, pipeline_predict_fn,num_features=len(self.feature_names))
                feature_weight = exp.as_map() #weigths given by lime for features
                lime_weights[label] = feature_weight

                end = time.perf_counter()
                average_time_lime[label] += end - start
        
        for label in self.pipelines:
            average_time_lime[label] /= self.n
        
        return average_time_lime,lime_weights
    
    def get_shap_explanations(self):
        average_time_shap = defaultdict(int)
        shap_values = defaultdict(int)

        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            k_explainer = shap.KernelExplainer(pipeline_predict_fn, self.X_train)
            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]
                start = time.perf_counter()
                shap_values[label] = label_shap_values = k_explainer.shap_values(row_to_explain)[1]
                end = time.perf_counter()
                average_time_shap[label] = end - start
        
        for label in self.pipelines:
            average_time_shap[label] /= self.n
        return average_time_shap,shap_values
    
    def get_anchor_explanation(self):
        # print("anchor")
        explainer = anchor_tabular.AnchorTabularExplainer(
            self.class_names,
            self.feature_names,
            self.X_train.values,
            self.categorical_names)

        average_time_anchor = defaultdict(int)
        for label,pipeline in self.pipelines.items():
            for i in range(self.n):
                row_to_explain = self.X_test.iloc[i]
                start = time.perf_counter()
                pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=False)
                exp = explainer.explain_instance(row_to_explain.values, pipeline_predict_fn, threshold=0.8)
                end = time.perf_counter()
                average_time_anchor[label] = end - start
        
        for label in self.pipelines:
            average_time_anchor[label] /= self.n
        return average_time_anchor 

    def get_average_explanation_times(self):
        average_times = {
            "lime" : self.get_lime_explanations(),
            "shap" : self.get_shap_explanations(),
            "anchor" : self.get_anchor_explanation(),
        }

        average_path = "Average_time"
        if not os.path.exists(average_path):
            os.makedirs(average_path)

        with open(f'{average_path}/{self.dataset_name}.json', "w") as outfile:
            json.dump(average_times, outfile)
        
        print(f"Done for {self.dataset_name}")
        return average_times
    
    def measure_monotonicity(self,pred_prob):
        n = len(pred_prob)
        inc,dec = 0,0
        for i in range(1,n):
            if pred_prob[i] > pred_prob[i - 1]:
                inc += 1
            elif pred_prob[i] < pred_prob[i - 1]:
                dec += 1
        return inc,dec
    
    def calculate_lime_faithfullness(self):
        lime_faithfulness = defaultdict(int)
        shap_faithfulness = defaultdict(int)

        lime_monotonicity = defaultdict(int)


        base = np.mean(self.X_train,axis=0)
        # _,lime_weights = self.get_lime_explanations()
        # _,shap_vals = self.get_shap_explanations()
        
        self.n = 20
        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            positive_count = 0
            lime_inc,lime_dec = 1,1

            for ind in range(self.n):
                x = self.X_test.iloc[ind]
                pred_class = np.argmax(pipeline_predict_fn(x), axis=1)[0]
                if pred_class == 1:
                    positive_count += 1
                    #mean values to simulate removing a feature
                    #use modes instead of mean for categorical data to simulate removing
                    for categorical in self.categorical_features:
                        categorical_index = self.feature_names.index(categorical)
                        unique_vals,counts = np.unique(self.X_train[categorical], return_counts=True)
                        mode_index = np.argmax(counts)
                        base[categorical_index] = mode_index
                        
                    
                    label_lime_weights = self.get_lime_explanation(ind,pipeline_predict_fn)
                    # print("RF prediction",pipeline_predict_fn(x))
                    # print("Prediction class",np.argmax(pipeline_predict_fn(x), axis=1)[0])               
                    #lime faithfulness
                    #find indexs of coefficients in decreasing order of value
                    ar = np.argsort(-label_lime_weights)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
                    positive_weight_index = []
                    pred_probs = np.zeros(x.shape[0])
                    monotonicity_probs = []

                    for ind in ar:
                        if label_lime_weights[ind] >= 0:
                            positive_weight_index.append(ind)
                        x_copy = x.copy(deep=True)
                        x_copy.iloc[ind] = base.iloc[ind]
                        x_copy_pr = pipeline_predict_fn(x_copy)
                        pred_probs[ind] = x_copy_pr[0][pred_class]
                        monotonicity_probs.append(pred_probs[ind])
                        # print(label_lime_weights[ind],pred_probs[ind])

                    positive_weight_predictions = [0] * len(positive_weight_index)
                    positive_weights = [0] * len(positive_weight_index) 

                    for ind,idx in enumerate(positive_weight_index):
                        positive_weight_predictions[ind] = pred_probs[idx]
                        positive_weights[ind] = label_lime_weights[idx]

                    # print(positive_weights)
                    # print(positive_weight_predictions)
                    # print(-np.corrcoef(positive_weights, positive_weight_predictions)[0,1])
                    # return
                    lime_faithfulness[label] += -np.corrcoef(label_lime_weights, pred_probs)[0,1] 

                    increase,decrease = self.measure_monotonicity(monotonicity_probs)
                    lime_inc += increase
                    lime_dec += decrease
                    # print(label,np.corrcoef(range(x.shape[0]), pred_probs)[0,1])
                    # print(label_lime_weights)
                    # print(pred_probs)
                    


                    # #shap faithfulness
                    # label_shap_vals = shap_vals[label]
                    # label_shap_vals = np.array(list(filter(lambda x: x > 0, label_shap_vals)))

                    # ar = np.argsort(-label_shap_vals)
                    # pred_probs = np.zeros(label_shap_vals.shape[0])
                    # for ind in ar:
                    #     x_copy = x.copy(deep=True)
                    #     x_copy.iloc[ind] = base.iloc[ind]
                    #     x_copy_pr = pipeline_predict_fn(x_copy)
                    #     pred_probs[ind] = x_copy_pr[0][pred_class]

                    # corr_coff = np.corrcoef(label_shap_vals,pred_probs)[0,1]
                    # if np.isnan(corr_coff):
                    #     corr_coff = 0
                    # shap_faithfulness[label] += corr_coff / self.n
                    # print(label,shap_faithfulness[label])
                    # # print(pred_probs)
            
            lime_faithfulness[label] /= positive_count
            lime_monotonicity[label] = lime_inc / (lime_inc + lime_dec)
        # print(lime_faithfulness)

        print(lime_monotonicity)
        return lime_faithfulness,shap_faithfulness
    
    # /Users/kiduswondimgagegnehu/Documents/IT/AURA/Running_Time

        
        



    

        



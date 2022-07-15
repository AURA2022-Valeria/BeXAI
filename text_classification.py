from collections import defaultdict
import json
from operator import index
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
import spacy

import xgboost as xgb


from lime.lime_text import LimeTextExplainer
import shap
from anchor import anchor_text

import pickle
import os
import time

from classification import Classification

class TextClassification(Classification):
    def __init__(self,dataset_name,X,Y,class_names):
        super().__init__(dataset_name,X,Y,class_names)
        self.tfidf = TfidfVectorizer()
        self.nlp = spacy.load('en_core_web_sm')
        self.X_train_vectorized = self.tfidf.fit_transform(self.X_train)
        self.X_test_vectorized  = self.tfidf.transform(self.X_test)
        self.n = min(3,self.X_test.shape[0])

        self._load_models()

    def _load_models(self):
        file_path = f'models/{self.dataset_name}'

        for label,model in self.models.items():
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            model_file_path = f'{file_path}/{label}' 
            try:
                self.models[label] = pickle.load(open(model_file_path, "rb")) 
            except:
                self.models[label] = model.fit(self.X_train_vectorized,self.Y_train)
                pickle.dump(self.models[label], open(model_file_path, "wb")) #save the trained model inorder not to train it again
                
    def get_explanations(self,output=False):
        """
        Explains blackbox models and measure the time taken for the explainers
        """
        #TODO: modularize the time measurement 
        #TODO: Should graph representation be included in time measurements?
        index_to_explain = 10
        timer = {
            "lime" : {},
            "shap" : {},
            "anchor" : {},
        }
        
        row_to_explain = self.X_test[index_to_explain]  

        #lime explanation
        explainer = LimeTextExplainer(class_names=self.class_names)
        for label,model in self.models.items():
            pipeline = make_pipeline(self.tfidf,model)
            exp = explainer.explain_instance(row_to_explain, pipeline.predict_proba, num_features=6,top_labels = 1)
            feature_weight = exp.as_map() #weigths given by lime for words

            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                exp.save_to_file(f'{path}/lime_{label}.html')
        
        #shap explanation
        row_to_explain = self.X_test_vectorized[index_to_explain]

        sampling_time_start = time.perf_counter()
        X_train_sample = shap.kmeans(self.X_train_vectorized, 30)
        sampling_time_end = time.perf_counter()
        timer["shap_sampling_time"] = sampling_time_end - sampling_time_start

        for label,model in self.models.items():
            predict_fn = lambda X: model.predict(X)
            SHAP_explainer = shap.KernelExplainer(predict_fn, X_train_sample)
            shap_vals = SHAP_explainer.shap_values(row_to_explain,l1_reg="num_features(5)")
            colour_test = pd.DataFrame(row_to_explain.todense())
            figure = shap.force_plot(SHAP_explainer.expected_value, shap_vals, 
                colour_test.iloc[0,:], feature_names=self.tfidf.get_feature_names(),show=False,matplotlib = True)

            if output == True:
                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                figure.savefig(f'{path}/shap_{label}.png')
        
        explainer = anchor_text.AnchorText(self.nlp, self.class_names, use_unk_distribution=True)
        for label,model in self.models.items():
            predict_fn = self._get_anchor_prediction_fn(model)
            text = self.X_test[index_to_explain]
            pred = explainer.class_names[predict_fn([text])[0]]
            exp = explainer.explain_instance(text, predict_fn, threshold=0.3)
        
            if output == True:
                path = f'Explanations/{self.dataset_name}'
                try:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    exp.save_to_file(f'{path}/anchor_{label}.html')
                except:
                    print("Couldn't generate html")
                    print(label)
                    print(" AND ".join(exp.names()))
                    print("Precision:",exp.precision())
                    print("Coverage:",exp.coverage())


                
    def get_average_explanation_lime(self):
        average_time_lime = defaultdict(int)

        for i in range(self.n):
            row_to_explain = self.X_test[i]
            explainer = LimeTextExplainer(class_names=self.class_names)
            for label,model in self.models.items():
                start = time.perf_counter()
                pipeline = make_pipeline(self.tfidf,model)
                exp = explainer.explain_instance(row_to_explain, pipeline.predict_proba, num_features=6,top_labels = 3)
                feature_weight = exp.as_map() #weigths given by lime for words
                end = time.perf_counter()
                average_time_lime[label] = end - start
        
        for label in self.models:
            average_time_lime[label] /= self.n
        return average_time_lime
    
    def get_average_explanation_shap(self):
        average_time_shap = defaultdict(int)

        sampling_time_start = time.perf_counter()
        X_train_sample = shap.kmeans(self.X_train_vectorized, 30)
        sampling_time_end = time.perf_counter()
        sampling_time = sampling_time_end - sampling_time_start


        for label,model in self.models.items(): 
            for i in range(self.n):
                row_to_explain = self.X_test_vectorized[i]
                start = time.perf_counter()
                predict_fn = lambda X: model.predict(X)
                SHAP_explainer = shap.KernelExplainer(predict_fn, X_train_sample)
                shap_vals = SHAP_explainer.shap_values(row_to_explain,l1_reg="num_features(5)")
                colour_test = pd.DataFrame(row_to_explain.todense())
                figure = shap.force_plot(SHAP_explainer.expected_value, shap_vals, 
                    colour_test.iloc[0,:], feature_names=self.tfidf.get_feature_names(),show=False,matplotlib = True)
                end = time.perf_counter()
                average_time_shap[label] = end - start
        
        for label in self.models:
            average_time_shap[label] /= self.n
            average_time_shap[label] += sampling_time
        
        average_time_shap["smapling_time"] = sampling_time
        return average_time_shap
    
    def _get_anchor_prediction_fn(self,model):
        def predict_fn(text):
            return model.predict(self.tfidf.transform(text))
        return predict_fn

    def get_average_explanation_anchor(self):
        average_time_anchor = defaultdict(int)
        explainer = anchor_text.AnchorText(self.nlp, self.class_names, use_unk_distribution=True)
        for label,model in self.models.items():
            for i in range(self.n):
                start = time.perf_counter()
                predict_fn = self._get_anchor_prediction_fn(model)
                text = self.X_test[i]
                pred = explainer.class_names[predict_fn([text])[0]]
                exp = explainer.explain_instance(text, predict_fn, threshold=0.3)
                end = time.perf_counter()
                average_time_anchor[label] = end - start
        for label in self.models:
            average_time_anchor[label] /= self.n
        return average_time_anchor

    
    def get_average_explanation_times(self):
        average_times = {
            "lime" : self.get_average_explanation_lime(),
            "shap" : self.get_average_explanation_shap(),
            "anchor" : self.get_average_explanation_anchor(),
        }

        average_path = "running_Time/"
        if not os.path.exists(average_path):
            os.makedirs(average_path)

        with open(f'{average_path}{self.dataset_name}.json', "w") as outfile:
            json.dump(average_times, outfile)
        return average_times

        



            

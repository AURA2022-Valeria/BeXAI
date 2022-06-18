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



#class Dataset(): #abstract class

# class RegressionDataset(Dataset):


class Classification:
    def __init__(self,dataset_name,X,Y,class_names):
        """
        dataset_name : unique name to the dataset
        X : features of the dataset
        Y : column to be predicted
        class_names : name of classes on classification dataset
        feature_names : Name of columns in the dataset used to train models
        categorical features : 
        """
        self.dataset_name = dataset_name
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X, Y,test_size=0.3, random_state=0)
        self.class_names = class_names
        self.models = {"random_forest" : RandomForestClassifier(), 
                        "knn" : KNeighborsClassifier() , 
                        # "MLP" : MLPClassifier(hidden_layer_sizes=(120, 120,120), activation='relu', solver='lbfgs', max_iter=10000,alpha=1e-5,tol=1e-5), 
                        "decision_tree" : DecisionTreeClassifier(), 
                        "xgboost" : xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)
                        }



class TextClassification(Classification):
    def __init__(self,dataset_name,X,Y,class_names):
        super().__init__(dataset_name,X,Y,class_names)
        self.tfidf = TfidfVectorizer()
        self.X_train_vectorized = self.tfidf.fit_transform(self.X_train)
        self.X_test_vectorized  = self.tfidf.transform(self.X_test)
        self.load_models()

    def load_models(self):
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
                
    def get_explanations(self):
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
        
        row_to_explain = self.X_train[index_to_explain]  

        #lime explanation
        explainer = LimeTextExplainer(class_names=self.class_names)
        for label,model in self.models.items():
            start = time.perf_counter()
            pipeline = make_pipeline(self.tfidf,model)
            exp = explainer.explain_instance(row_to_explain, pipeline.predict_proba, num_features=6,top_labels = 3)
            feature_weight = exp.as_map() #weigths given by lime for words
            end = time.perf_counter()
            timer["lime"][f'{self.dataset_name} - {label}'] = end - start

            path = f'Explanations/{self.dataset_name}'
            if not os.path.exists(path):
                os.makedirs(path)
            exp.save_to_file(f'{path}/lime_{label}.html')
        
        #shap explanation
        
        
        #save the time it took to explain in a json file
        with open(f"Running_Time/explanation_time_{self.dataset_name}.json", "w") as outfile:
            json.dump(timer, outfile)



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
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(X.iloc[:,X.columns.get_loc(feature)])
            self.categorical_names[self.categorical_features_indexes[i]] = list(le.classes_)
            X[feature] = list(le.transform(X[feature]))
        
        
        self.encoder = ColumnTransformer(
            [('OHE', OneHotEncoder(),categorical_features)],
            remainder = 'passthrough'
            ) 
        self.encoder.fit(X)

        #X is to be splitted after the label encoding steps 
        super().__init__(dataset_name,X,Y,class_names)

        #pipeline using the encoder and ml_algorithms 
        self.pipelines = {}

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
            if probability:
                return pipeline.predict(pdf)
            return pipeline.predict_proba(pdf)
        return format_and_predict

    def get_explanations(self):
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

        row_to_explain = self.X_test.iloc[index_to_explain]
        
        #lime explanation
        lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values ,class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features_indexes, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False)
        
        for label,pipeline in self.pipelines.items():
            start = time.perf_counter()
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline)
            exp = lime_explainer.explain_instance(row_to_explain.values, pipeline_predict_fn)
            feature_weight = exp.as_map() #weigths given by lime for features
            end = time.perf_counter()
            timer["lime"][f'{self.dataset_name} - {label}'] = end - start

            path = f'Explanations/{self.dataset_name}'
            if not os.path.exists(path):
                os.makedirs(path)
            exp.save_to_file(f'{path}/lime_{label}.html')
    

        #shap explanation
        for label,pipeline in self.pipelines.items():
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline)
            k_explainer = shap.KernelExplainer(pipeline_predict_fn, self.X_train)
            start = time.perf_counter()
            shap_values = k_explainer.shap_values(row_to_explain)
            end = time.perf_counter()
            timer["shap"][f'{self.dataset_name} - {label}'] = end - start

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
            pipeline_predict_fn = self.get_pipeline_predicfn(pipeline,probability=True)
            exp = explainer.explain_instance(row_to_explain.values, pipeline_predict_fn, threshold=0.8)
            print(f"{label} Anchor explanation")
            print('Anchor: %s' % (' AND '.join(exp.names())))
            print('Precision: %.2f' % exp.precision())
            print('Coverage: %.2f' % exp.coverage())

        #save the time it took to explain in a json file
        with open(f"Running_Time/explanation_time_{self.dataset_name}.json", "w") as outfile:
            json.dump(timer, outfile)
        

        #anchor explanation

    

            
            
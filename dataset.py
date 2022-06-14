from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lime
from lime import lime_tabular
import pickle
import os
import shap
import time


#class Dataset(): #abstract class

# class RegressionDataset(Dataset):


class Classification_Dataset:
    #TODO : divide the class to regreesion and classification datasets 
    #TODO : the test_size can be passed as a parameter. should it?
    
    def __init__(self,dataset_name,X,Y,type,feature_names,class_names,categorical_features=[]):
        """
        dataset_name : unique name to the dataset
        X : features of the dataset
        Y : column to be predicted
        type: Type of the dataset - Tabular,Text or image
        class_names : name of classes on classification dataset
        feature_names : Name of columns in the dataset used to train models
        categorical features : 
        """
        self.dataset_name = dataset_name
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X, Y,test_size=0.2, random_state=0)        
        self.type = type
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features
        self.models = {}
        self.load_models() #dictionary of trained models with their label
    
    def load_models(self):
        ml_algorithms = {"random_foest" : RandomForestClassifier(), 
                        "knn" : KNeighborsClassifier() , 
                        "MLP" : MLPClassifier(hidden_layer_sizes=(120, 120,120, 120), activation='relu', solver='lbfgs', max_iter=10000,alpha=1e-5,tol=1e-5), 
                        "decision_tree" : DecisionTreeClassifier(), 
                        "xgboost" : xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)}
        file_path = f'Models/{self.dataset_name}'

        for label,ml_algorithm in ml_algorithms.items():          
            try:
                self.models[label] = pickle.load(open(file_path, "rb"))
            except:
                self.models[label] = ml_algorithm.fit(self.X_train,self.Y_train)
                pickle.dump(self.models[label], open(file_path, "wb"))

        # for label,model in self.models.items():
        #     try:
        #         print(model.score(self.X_test,self.Y_test))
        #     except Exception as e:
        #         print(e)

        
    def get_explanations(self):
        #TODO: modularize the time measurement 
        #TODO: Should graph representation be included in time measurements?
        index_to_explain = 10
        timer = {
            "lime" : {},
            "shap" : {},
            "anchor" : {},
        }

        if self.type == "Tabular":
            row_to_explain = self.X_test.iloc[index_to_explain]
           
            #lime explanation
            lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values,class_names=self.class_names,feature_names=self.feature_names,discretize_continuous=True)
            for label,model in self.models.items():
                start = time.time()
                exp = lime_explainer.explain_instance(row_to_explain.values, model.predict_proba,top_labels=1)
                feature_weight = exp.as_map()
                end = time.time()
                timer["lime"][f'{self.dataset_name} - {label}'] = end - start

                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                exp.save_to_file(f'{path}/lime_{label}.html')
        

            #shap explanation
            k_explainer = shap.KernelExplainer(model.predict_proba, self.X_train)
            for label,model in self.models.items():
                start = time.time()
                shap_values = k_explainer.shap_values(row_to_explain)
                end = time.time()
                timer["shap"][f'{self.dataset_name} - {label}'] = end - start

                path = f'Explanations/{self.dataset_name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                shap.initjs()
                figure = shap.force_plot(k_explainer.expected_value[1], shap_values[1], row_to_explain,matplotlib = True, show = False)
                figure.savefig(f'{path}/shap_{label}.png')

        return timer
            
            
            
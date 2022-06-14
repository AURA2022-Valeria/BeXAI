import pandas as pd
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import shap
import lime

data_set_URL = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
data = pd.read_csv(data_set_URL)
data.variety.value_counts()
class_names = ["Setosa","Versicolor","Virginica"]


#encode categorical data
type_dict = {"variety" : {"Setosa" : 0, "Versicolor" : 1,"Virginica" : 2}}
data = data.replace(type_dict)

features = list(data.columns)
features.remove("variety")
X = data[features]
Y = data["variety"]
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = 0, test_size=0.35)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(120, 120,120, 120), activation='relu', solver='lbfgs', max_iter=10000,alpha=1e-5,tol=1e-5)
nn.fit(X_train.values,y_train)


#shap explanation
row_to_explain = X_test.iloc[20].values
#NN model explanation using kernel shap
k_explainer = shap.KernelExplainer(nn.predict_proba, X_train)
k_shap_values = k_explainer.shap_values(row_to_explain)
print(k_shap_values)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], row_to_explain)
# import pandas as pd
# import sklearn
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import xgboost as xgb
# from sklearn.metrics import accuracy_score
# import shap
# import pickle
# import matplotlib as plt

# data_set_URL = "./datasets/reddit.csv"
# data = pd.read_csv(data_set_URL)
# class_names = ["Physics","Chemistry","Biology"]
# #encode categorical data
# type_dict = {"Topic" : {"Physics" : 0, "Chemistry" : 1,"Biology" : 2}}
# data = data.replace(type_dict)
# data.head()

# X_raw = data['Comment'].values
# y_raw = data['Topic'].values
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw,y_raw,random_state = 0, test_size=0.30,shuffle = True)


# tfidf = TfidfVectorizer()
# X_train_tfidf = tfidf.fit_transform(X_train_raw)
# X_test_tfidf = tfidf.transform(X_test_raw)

# X_train, X_test = X_train_tfidf,X_test_tfidf


# rf = pickle.load(open("./models/Reddit/random_forest", "rb")) 

# # sampling data from the training and test set to reduce time-taken
# X_train_sample = shap.kmeans(X_train, 100)
# X_test_sample = X_test[20]
# print(X_test_sample)

# predict_fn = lambda X: rf.predict(X)

# # creating the KernelExplainer using the logistic regression model and training sample
# SHAP_explainer = shap.KernelExplainer(predict_fn, X_train_sample)
# # calculating the shap values of the test sample using the explainer 
# shap_vals = SHAP_explainer.shap_values(X_test_sample,l1_reg="num_features(5)")

# # converting the test samples to a dataframe 
# # this is necessary for non-tabular data in order for the visualisations 
# # to include feature value
# colour_test = pd.DataFrame(X_test_sample.todense())

# shap.initjs()


# # print(SHAP_explainer.expected_value.shape)
# # print(SHAP_explainer.expected_value)
# # print(len(shap_vals))
# # print(shap_vals)
# figure = shap.force_plot(SHAP_explainer.expected_value, shap_vals, 
#                 colour_test.iloc[0,:], feature_names=tfidf.get_feature_names(),show=False,matplotlib = True)

# # print("Expected value:",SHAP_explainer.expected_value)
# # print("Shap value:",shap_vals)#,len(shap_vals[0]),len(shap_vals))
# figure.savefig('shap_text_1.png')

# # print("Prediction:",rf.predict(X_test_sample))



# # Expected value: [0.43692916 0.22763225 0.33543859]
# # Shap value: 15325
# # Prediction: [2]
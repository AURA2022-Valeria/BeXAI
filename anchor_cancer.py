# import matplotlib.pyplot as plt
# import os
# import time
# import pandas as pd
# import matplotlib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from lime.lime_tabular import LimeTabularExplainer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb
# from alibi.explainers import AnchorTabular

# data = pd.read_csv("./datasets/cancer.csv")
# # class_names=["Malignant","Benign"]

# data=data.drop(columns=["id"])
# data=data.drop('Unnamed: 32', axis=1)
# print(data.shape)
# # exit()
# X=data.drop(columns=['diagnosis'])
# Y = data['diagnosis'].map({'B':0,'M':1})
# # X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=0.7)
# print(Y.shape)
# X_train=X[0:145]
# print("Hello")
# Y_train=[]
# for i in range(145):
#     Y_train.append(Y[i])
# Y_train=pd.DataFrame(Y_train)



# feature_names = list(X_train.columns)
# class_names=["Bendgn","Maligngant"]
# model=RandomForestClassifier(random_state=0)
# model.fit(X_train,Y_train)

# predict_fn = lambda x: model.predict_proba(x)
# start=time.perf_counter()
# explainer = AnchorTabular(predict_fn, feature_names)
# print("doifghodfgh")
# z = X_train.to_numpy()
# explainer.fit(z, disc_perc=(25, 50, 75))


# idx = 0
# print('Prediction: ', class_names[explainer.predictor(X_train.iloc[[idx],:].values.reshape(1, -1))[0]])
# print("He;;o")

# explanation = explainer.explain(X_train.iloc[[idx],:].values.reshape(1,-1), threshold=0.95)
# print('Anchor: %s' % (' AND '.join(explanation.anchor)))
# print('Precision: %.2f' % explanation.precision)
# print('Coverage: %.2f' % explanation.coverage)
# end= time.perf_counter()
# print(end-start," This is the total time it took")
# exit()
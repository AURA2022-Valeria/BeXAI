from collections import defaultdict
import json
import os
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.segmentation import mark_boundaries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost
from skimage.color import gray2rgb, rgb2gray, label2rgb
import matplotlib.pyplot as plt
import shap


mnist = fetch_openml('mnist_784')
images = mnist.data.to_numpy()

models = {
    "random_forest" : RandomForestClassifier(), 
    "knn" : KNeighborsClassifier() , 
    "MLP" : MLPClassifier(hidden_layer_sizes=(120, 120,120), activation='relu', solver='lbfgs', max_iter=10000,alpha=1e-5,tol=1e-5), 
    "decision_tree" : DecisionTreeClassifier(), 
    "xgboost" : xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)
}

img_rows, img_cols = 28, 28
def runShap():

    X_train, X_test, Y_train, Y_test = train_test_split(mnist.data.values, list(map(int,mnist.target.values)), stratify = mnist.target, test_size=0.3, random_state=0)

    for label,model in models.items():
        model.fit(X_train,Y_train)

    number_of_images_to_explain = 2

    #sample images - because using the whole training dataset to calculcate shap values is very slow
    number_of_samples = 3
    sampled_images = X_train[np.random.choice(len(X_train), number_of_samples, replace=False)]


    shap_average = defaultdict(int)
    for label,model in models.items():
        SHAP_explainer = shap.KernelExplainer(model.predict, sampled_images)
        for i in range(number_of_images_to_explain):
            start = time.perf_counter()
            shap_vals = SHAP_explainer.shap_values(X_train[i]).reshape(28,28,1)
            image_reshapped = np.array(X_train[i]).reshape(28,28,1)
            shap.image_plot(shap_vals,image_reshapped)
            end = time.perf_counter()
            shap_average[label] += end - start

    for label in models:
        shap_average[label] /= number_of_images_to_explain

    print(shap_average)
    # average_path = ""
    # if not os.path.exists(average_path):
    #     os.makedirs(average_path)
    #
    # with open(f'{average_path}/mnist_shap.json', "w") as outfile:
    #     json.dump(shap_average, outfile)

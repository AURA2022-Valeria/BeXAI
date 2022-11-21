class PipeStep(object):

    def __init__(self, step_func):
        self._step_func = step_func

    def fit(self, *args):
        return self

    def transform(self, X):
        return self._step_func(X)


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb  # since the code wants color images
from keras.datasets import mnist
from skimage.segmentation import mark_boundaries
import time
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def runLime():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # We make convert each image to rgb one so that Lime works properly
    X_train = np.stack([gray2rgb(iimg) for iimg in X_train.reshape((-1, 28, 28))], 0).astype(np.uint8)
    Y_train = Y_train.astype(np.uint8)

    X_test = np.stack([gray2rgb(iimg) for iimg in X_test.reshape((-1, 28, 28))], 0).astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)

    makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
    # we flatten the image to 1d to train them on a model

    flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

    simple_rf_pipeline = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('RF', RandomForestClassifier())
    ])

    simple_knn = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),
        ('RF', KNeighborsClassifier())
    ])

    simple_mlp = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('RF', MLPClassifier())
    ])
    simple_tree = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('RF', DecisionTreeClassifier())
    ])
    simple_xgb = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('xgb', xgb.XGBClassifier())
    ])

    simple_rf_pipeline.fit(X_train, Y_train)
    simple_mlp.fit(X_train, Y_train)
    simple_knn.fit(X_train, Y_train)
    simple_tree.fit(X_train, Y_train)
    simple_xgb.fit(X_train, Y_train)

    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    lime_dict = {}
    start = time.perf_counter()
    length = 15
    for i in range(length):
        explanation = explainer.explain_instance(X_test[i],
                                                 classifier_fn=simple_rf_pipeline.predict_proba,
                                                 top_labels=10, hide_color=0, num_samples=10000,
                                                 segmentation_fn=segmenter)
        temp, mask = explanation.get_image_and_mask(Y_test[i], positive_only=True, num_features=10, hide_rest=False,
                                                    min_weight=0.01)
        #
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
        plt.imshow(X_test[i])
        plt.show()
        fig, (ax1) = plt.subplots(1, figsize=(8, 4))
        ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
        ax1.set_title('Positive Regions for {}'.format(Y_test[i]))
        plt.show()
    end = time.perf_counter()
    average = (end - start) / length
    lime_dict['RandomForest'] = average

    temp, mask = explanation.get_image_and_mask(Y_test[0], positive_only=True, num_features=10, hide_rest=False,
                                                min_weight=0.01)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    plt.imshow(X_test[0])
    plt.show()

    fig, (ax1) = plt.subplots(1, figsize=(8, 4))
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(Y_test[0]))
    plt.show()
    start = time.perf_counter()
    for i in range(length, length + 30):
        explanation_knn = explainer.explain_instance(X_test[i],
                                                     classifier_fn=simple_knn.predict_proba,
                                                     top_labels=10, hide_color=0, num_samples=10000,
                                                     segmentation_fn=segmenter)
    end = time.perf_counter()

    average = (end - start) / length
    lime_dict["knn"] = average
    temp, mask = explanation_knn.get_image_and_mask(Y_test[0], positive_only=True, num_features=10, hide_rest=False,
                                                    min_weight=0.01)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    plt.imshow(X_test[0])
    plt.show()

    fig, (ax1) = plt.subplots(1, figsize=(8, 4))
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(Y_test[0]))
    plt.show()
    start = time.perf_counter()
    for i in range(5):
        explanation_mlp = explainer.explain_instance(X_test[i],
                                                     classifier_fn=simple_mlp.predict_proba,
                                                     top_labels=10, hide_color=0, num_samples=10000,
                                                     segmentation_fn=segmenter)
    end = time.perf_counter()
    average = (end - start) / length
    lime_dict["mlp"] = average

    temp, mask = explanation_mlp.get_image_and_mask(Y_test[0], positive_only=True, num_features=10, hide_rest=False,
                                                    min_weight=0.01)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    plt.imshow(X_test[0])
    plt.show()

    fig, (ax1) = plt.subplots(1, figsize=(8, 4))
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(Y_test[0]))
    plt.show()
    start = time.perf_counter()

    for i in range(length):
        explanation_tree = explainer.explain_instance(X_test[i],
                                                      classifier_fn=simple_tree.predict_proba,
                                                      top_labels=10, hide_color=0, num_samples=10000,
                                                      segmentation_fn=segmenter)

    end = time.perf_counter()
    average = (end - start) / length
    lime_dict["DecsionTree"] = average

    temp, mask = explanation_tree.get_image_and_mask(Y_test[0], positive_only=True, num_features=10, hide_rest=False,
                                                     min_weight=0.01)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    plt.imshow(X_test[0])
    plt.show()

    fig, (ax1) = plt.subplots(1, figsize=(8, 4))
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(Y_test[0]))
    plt.show()

    start = time.perf_counter()

    for i in range(length):
        explanation_tree = explainer.explain_instance(X_test[i],
                                                      classifier_fn=simple_xgb.predict_proba,
                                                      top_labels=10, hide_color=0, num_samples=10000,
                                                      segmentation_fn=segmenter)

    end = time.perf_counter()
    average = (end - start) / length
    lime_dict["xgb"] = average

    print(lime_dict)

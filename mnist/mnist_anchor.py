
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from alibi.explainers import AnchorImage
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
from keras.datasets import mnist
import xgboost as xgb
from skimage.segmentation import mark_boundaries
from sklearn.model_selection import train_test_split
from lime.wrappers.scikit_image import SegmentationAlgorithm
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import time

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

def runAnchor():

    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()

    X_train = np.stack([gray2rgb(iimg) for iimg in X_train.reshape((-1, 28, 28))],0).astype(np.uint8)
    Y_train = Y_train.astype(np.uint8)

    X_test = np.stack([gray2rgb(iimg) for iimg in X_test.reshape((-1, 28, 28))],0).astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)







    makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
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

    simple_mlp=Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('RF', MLPClassifier())
                                  ])
    simple_tree=Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('RF', DecisionTreeClassifier())
                                  ])
    simple_xgb=Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),

        ('xgb', xgb.XGBClassifier())
                                  ])

    anchor_dict={}

    simple_rf_pipeline.fit(X_train, Y_train)
    simple_knn.fit(X_train,Y_train)
    simple_mlp.fit(X_train,Y_train)
    simple_tree.fit(X_train,Y_train)
    simple_xgb.fit(X_train,Y_train)


    segmentation_fn = 'slic'
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}
    # print(X_train[0])
    # print(Y_train[0])
    # print(Y_train[1])
    # print(X_test[0].shape)
    start=time.perf_counter()
    length=5
    explainer = AnchorImage(simple_rf_pipeline.predict_proba, X_test[20].shape, segmentation_fn=segmenter,
                            segmentation_kwargs=kwargs, images_background=None)
    for i in range(length):
        explanation = explainer.explain(X_test[i], threshold=.95, p_sample=.5, tau=0.25)
        # print(explanation.anchor)
        plt.imshow(X_test[i])
        plt.show()
        plt.imshow(explanation.anchor[:,:,0])
        plt.show()
    end =time.perf_counter()
    average = (end-start)/(length)
    anchor_dict['RandomFores']=average
    # print(average," This is random Forest")



    start=time.perf_counter()
    explainer = AnchorImage(simple_knn.predict_proba, X_test[20].shape, segmentation_fn=segmenter,
                            segmentation_kwargs=kwargs, images_background=None)
    #print("four")
    for i in range(length):
        explanation = explainer.explain(X_test[i], threshold=.95, p_sample=.5, tau=0.25)
        # print(explanation.anchor)
        plt.imshow(X_test[i])
        plt.show()
        plt.imshow(explanation.anchor[:,:,0])
        plt.show()
    end =time.perf_counter()
    average = (end-start)/(length)
    anchor_dict['knn']=average
    # print(average,"   This is for knn")
    start=time.perf_counter()
    explainer = AnchorImage(simple_mlp.predict_proba, X_test[20].shape, segmentation_fn=segmenter,
                            segmentation_kwargs=kwargs, images_background=None)

    for i in range(length):
        explanation = explainer.explain(X_test[i], threshold=.95, p_sample=.5, tau=0.25)
        # print(explanation.anchor)
        plt.imshow(X_test[i])
        plt.show()
        plt.imshow(explanation.anchor[:,:,0])
        plt.show()
    end =time.perf_counter()
    average = (end-start)/(length)
    anchor_dict['mlp']=average
    # print(average,"   This is for mlp")




    start=time.perf_counter()
    explainer = AnchorImage(simple_tree.predict_proba, X_test[20].shape, segmentation_fn=segmenter,
                            segmentation_kwargs=kwargs, images_background=None)
    #print("four")
    for i in range(length):
        explanation = explainer.explain(X_test[i], threshold=.95, p_sample=.5, tau=0.25)
        # print(explanation.anchor)
        # plt.imshow(X_test[i])
        # plt.show()
        # plt.imshow(explanation.anchor[:,:,0])
        # plt.show()
    end =time.perf_counter()
    average = (end-start)/(length)
    anchor_dict['DecsionTree']=average
    #print(average,"   This is for Decision Tree")



    start=time.perf_counter()
    explainer = AnchorImage(simple_xgb.predict_proba, X_test[20].shape, segmentation_fn=segmenter,
                            segmentation_kwargs=kwargs, images_background=None)
    # print("four")
    for i in range(length):
        explanation = explainer.explain(X_test[i], threshold=.95, p_sample=.5, tau=0.25)
        #print(explanation.anchor)
        plt.imshow(X_test[i])
        plt.show()
        plt.imshow(explanation.anchor[:,:,0])
        plt.show()
    end =time.perf_counter()
    average = (end-start)/(length)
    anchor_dict['xgb']=average
    #print(average,"   This is for xgb ")


    print(anchor_dict)


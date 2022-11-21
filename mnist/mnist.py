import pandas as pd
from mnist.mnist_anchor import runAnchor
from mnist.mnist_lime import runLime
from mnist.mnist_shap import runShap
from sklearn.model_selection import train_test_split

def mnist_runtime():
    print("Anchor mnist runtime")
    runAnchor()
    print("Lime mnist runtime")
    runLime()
    print("Shap mnist runtime")
    runShap()
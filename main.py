from sklearn.datasets import load_wine
import pandas as pd
from dataset import Classification_Dataset

#wine_dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_dataset = Classification_Dataset(dataset_name="wine",X=df,Y=wine_data.target,type="Tabular",feature_names=wine_data.feature_names,class_names=list(wine_data.target_names))
# wine_dataset.get_explanations()

#iris_flower
data = pd.read_csv("./datasets/iris.csv")
iris_class_names = ["Setosa","Versicolor","Virginica"]
iris_features = list(data.columns)
prediction_feature = "variety"
iris_features.remove(prediction_feature)
iris_dataset = Classification_Dataset(dataset_name="iris",X=data[iris_features],Y=data[prediction_feature],type="Tabular",feature_names=iris_features,class_names=iris_class_names)
iris_timer = iris_dataset.get_explanations()

print(iris_timer)

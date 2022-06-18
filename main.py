from sklearn.datasets import load_wine,load_boston,load_breast_cancer
import pandas as pd
from dataset import TabularClassification,TextClassification
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# ##titanic_dataset
# data = pd.read_csv("./datasets/titanic.csv")

# #data clean up
# data['Deck'] = data['Cabin'].str.extract(r'([A-Z])?(\d)')[0]
# data['Age'] = data['Age'].fillna(data['Age'].mean())
# data['Deck'] = data['Deck'].fillna("0")

# data['Embarked'] = data['Embarked'].fillna('S') #use the most frequency pick up place
# data['Embarked'] = pd.Categorical(data['Embarked'], categories=['S', 'C', 'Q'])

# #drop unused columns
# to_remove = ["Name","PassengerId","Ticket","Cabin"]
# for col in to_remove:
#   data.drop(col,axis=1,inplace=True)

# features = list(data.columns) 
# features.remove('Survived')

# class_names = ["No","Yes"]
# categorical_features = ["Pclass","Sex","Parch","Embarked","Deck"]

# titanic_dataset = TabularClassification(dataset_name="Titanic",X=data[features],Y=data.Survived,feature_names=features,class_names=class_names,categorical_features=categorical_features)
# titanic_dataset.get_explanations()


#wine_dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_dataset = TabularClassification(dataset_name="wine",X=df,Y=wine_data.target,feature_names=wine_data.feature_names,class_names=list(wine_data.target_names))
wine_dataset.get_explanations()


# #iris_flower
# data = pd.read_csv("./datasets/iris.csv")
# iris_type_dict = {"variety" : {"Setosa" : 0, "Versicolor" : 1,"Virginica" : 2}}
# data = data.replace(iris_type_dict)
# iris_class_names = ["Setosa","Versicolor","Virginica"]
# iris_features = list(data.columns)
# prediction_feature = "variety"
# iris_features.remove(prediction_feature)
# iris_dataset = TabularClassification(dataset_name="iris",X=data[iris_features],Y=data[prediction_feature],feature_names=iris_features,class_names=iris_class_names)
# iris_timer = iris_dataset.get_explanations()


# #reddit_dataset
# data = pd.read_csv("./datasets/reddit.csv")
# reddit_type_dict = {"Topic" : {"Physics" : 0, "Chemistry" : 1,"Biology" : 2}}
# data = data.replace(reddit_type_dict)
# reddit_class_names = ["Physics","Chemistry","Biology"]
# X = data['Comment'].values
# Y = data['Topic'].values
# reddit_dataset = TextClassification(dataset_name="Reddit",X=X,Y=Y,class_names=reddit_class_names)
# reddit_dataset.get_explanations()

#cancer_dataset
# data = pd.read_csv("./datasets/cancer.csv")
# data=data.drop(columns=["id"])
# data=data.drop('Unnamed: 32', axis=1)
# X=data.drop(columns=['diagnosis'])
# Y = data['diagnosis'].map({'B':0,'M':1})
# cancer_feature_names = list(X.columns)
# cancer_class_names=["Bendgn","Maligngant"]
# cancer_dataset = TabularClassification(dataset_name="Cancer",X=X,Y=Y,feature_names=cancer_class_names,class_names=cancer_class_names)
# cancer_dataset.get_explanations()


# cancer_data = load_breast_cancer()
# df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
# cancer_dataset = TabularClassification(dataset_name="Cancer",X=df,Y=cancer_data.target,feature_names=cancer_data.feature_names,class_names=list(cancer_data.target_names))
# cancer_dataset.get_explanations()
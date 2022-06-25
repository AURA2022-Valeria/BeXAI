from typing import Text
from sklearn.datasets import load_wine,load_boston,load_breast_cancer
import pandas as pd
from text_classification import TextClassification
from tabular_classification import TabularClassification
import numpy as np


# ##titanic_dataset
data = pd.read_csv("./datasets/titanic.csv")

#data clean up
data['Deck'] = data['Cabin'].str.extract(r'([A-Z])?(\d)')[0]
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Deck'] = data['Deck'].fillna("0")

data['Embarked'] = data['Embarked'].fillna('S') #use the most frequency pick up place
data['Embarked'] = pd.Categorical(data['Embarked'], categories=['S', 'C', 'Q'])

#drop unused columns
to_remove = ["Name","PassengerId","Ticket","Cabin"]
for col in to_remove:
  data.drop(col,axis=1,inplace=True)

features = list(data.columns) 
features.remove('Survived')

class_names = ["No","Yes"]
categorical_features = ["Pclass","Sex","Parch","Embarked","Deck"]

titanic_dataset = TabularClassification(dataset_name="Titanic",X=data[features],Y=data.Survived,feature_names=features,class_names=class_names,categorical_features=categorical_features)
# titanic_dataset.get_explanations(output=True)
titanic_dataset.get_average_explanation_times()


#cancer_dataset
data = pd.read_csv("./datasets/cancer.csv")
data = data.drop(columns=["id"])
data = data.drop('Unnamed: 32', axis=1)

X = data.drop(columns=['diagnosis'])
Y = data['diagnosis'].map({'B':0,'M':1})

class_names = ["Benign","Malignant"]
feature_names = list(X.columns)

cancer_dataset = TabularClassification(dataset_name="Cancer",X=X,Y=Y,feature_names=feature_names,class_names=class_names)
cancer_dataset.get_average_explanation_times()


# wine_dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_dataset = TabularClassification(dataset_name="wine",X=df,Y=wine_data.target,feature_names=wine_data.feature_names,class_names=list(wine_data.target_names))
wine_dataset.get_average_explanation_times()


# #iris_flower
# data = pd.read_csv("./datasets/iris.csv")
# iris_type_dict = {"variety" : {"Setosa" : 0, "Versicolor" : 1,"Virginica" : 2}}
# data = data.replace(iris_type_dict)
# iris_class_names = ["Setosa","Versicolor","Virginica"]
# iris_features = list(data.columns)
# prediction_feature = "variety"
# iris_features.remove(prediction_feature)
# iris_dataset = TabularClassification(dataset_name="iris",X=data[iris_features],Y=data[prediction_feature],feature_names=iris_features,class_names=iris_class_names)
# iris_dataset.get_average_explanation_times()


#loan
# data = pd.read_csv("./datasets/loan.csv")
# Y = data['Loan_Status'].map({"N":0,"Y":1})
# data.drop(["Loan_Status","Loan_ID"],axis=1,inplace=True)

# for col in data.columns:
#     if data[col].isnull().sum() != 0:
#         data[col].fillna(data[col].value_counts().idxmax(),inplace = True)

# loan_features = data.columns
# loan_categorical_features = [col for col in data.columns if data[col].dtype == "object"]
# loan_class_names = ["Loan_Denied","Loan_Approved"]
# loan_dataset = TabularClassification(dataset_name="loan",X=data,Y=Y,feature_names=data.columns,class_names=loan_class_names,categorical_features=loan_categorical_features)
# # loan_dataset.get_explanations(output=True)
# loan_dataset.get_average_explanation_times()

#reddit_dataset
# data = pd.read_csv("./datasets/reddit.csv")
# reddit_type_dict = {"Topic" : {"Physics" : 0, "Chemistry" : 1,"Biology" : 2}}
# data = data.replace(reddit_type_dict)
# reddit_class_names = ["Physics","Chemistry","Biology"]
# X = data['Comment'].values
# Y = data['Topic'].values
# reddit_dataset = TextClassification(dataset_name="Reddit",X=X,Y=Y,class_names=reddit_class_names)
# # reddit_dataset.get_explanations(output=True)
# reddit_dataset.get_average_explanation_times() 

#spam_ham
# data = pd.read_csv("./datasets/spam_ham.csv")
# spam_ham_dict = {"Label" : {"spam" : 0, "ham" : 1}}
# data = data.replace(spam_ham_dict)
# spam_ham_class_names = ["Spam","ham"]
# X = data['EmailText'].values
# Y = data['Label'].values
# spam_ham_dataset = TextClassification(dataset_name="spam_ham",X=X,Y=Y,class_names=spam_ham_class_names)
# spam_ham_dataset.get_average_explanation_times() 
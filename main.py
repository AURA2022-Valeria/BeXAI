from sklearn.datasets import load_wine
import pandas as pd
from dataset import TabularClassification,TextClassification


##titanic_dataset
data = pd.read_csv("./datasets/titanic.csv")

#data clean up
data['Deck'] = data['Cabin'].str.extract(r'([A-Z])?(\d)')[0]
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Deck'] = data['Deck'].fillna("0")

data['Embarked'] = data['Embarked'].fillna('S') #use the most frequency pick up place
data['Embarked'] = pd.Categorical(data['Embarked'], categories=['S', 'C', 'Q'])
df = pd.get_dummies(data, columns=["Pclass","Sex","Parch","Deck","Embarked"], prefix=["Pclass","Sex","Parch","Deck","Embarked"])

#drop unused columns
to_remove = ["Name","PassengerId","Ticket","Cabin"]
for col in to_remove:
  df.drop(col,axis=1,inplace=True)

features = list(df.columns) 
class_names = ["No","Yes"]
features.remove('Survived')

titanic_dataset = TabularClassification(dataset_name="Titanic",X=df[features],Y=df.Survived,feature_names=features,class_names=class_names,categorical_features=[])
titanic_dataset.get_explanations()



# #wine_dataset
# wine_data = load_wine()
# df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
# wine_dataset = TabularClassification(dataset_name="wine",X=df,Y=wine_data.target,feature_names=wine_data.feature_names,class_names=list(wine_data.target_names))
# wine_dataset.get_explanations()


# #iris_flower
# data = pd.read_csv("./datasets/iris.csv")
# iris_class_names = ["Setosa","Versicolor","Virginica"]
# iris_features = list(data.columns)
# prediction_feature = "variety"
# iris_features.remove(prediction_feature)
# iris_dataset = TabularClassification(dataset_name="iris",X=data[iris_features],Y=data[prediction_feature],feature_names=iris_features,class_names=iris_class_names)
# iris_timer = iris_dataset.get_explanations()


#reddit_dataset
# data = pd.read_csv("./datasets/reddit.csv")
# reddit_type_dict = {"Topic" : {"Physics" : 0, "Chemistry" : 1,"Biology" : 2}}
# data = data.replace(reddit_type_dict)
# reddit_class_names = ["Physics","Chemistry","Biology"]
# X = data['Comment'].values
# Y = data['Topic'].values
# reddit_dataset = TextClassification(dataset_name="Reddit",X=X,Y=Y,class_names=reddit_class_names)
# reddit_dataset.get_explanations()
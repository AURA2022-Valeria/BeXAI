from sklearn.datasets import load_wine,load_boston,load_breast_cancer
import pandas as pd
from text_classification import TextClassification
from tabular_classification import TabularClassification

import argparse
import sys


Tabular = ["titanic","cancer","iris","wine","diabetes","loan"]
Text = ["reddit"]
Image = ["mnist"]
datasets = Tabular + Text + Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="selects the dataset the evaluations are measured on",choices=datasets)
parser.add_argument("--evaluation", help="evaluate based on either fidelity or runtime. fidelity is implemented only for tabular datasets",choices=["fidelity","runtime"])
parser.add_argument("--explain",help="generates 3 graphical explanations using the 3 explainers for an index selected from the test data",type=int)

args = parser.parse_args()
if not args.dataset or args.dataset not in datasets:
    sys.exit("Invalid or none dataset selected")
selected_dataset_title = args.dataset

if args.evaluation and args.evaluation.lower() == "fidelity" and selected_dataset_title not in Tabular:
    sys.exit("Invalid metric for the selected dataset")


if selected_dataset_title == "cancer":
    # cancer_dataset
    data = pd.read_csv("./datasets/cancer.csv")
    data = data.drop(columns=["id"])
    data = data.drop('Unnamed: 32', axis=1)
    X = data.drop(columns=['diagnosis'])
    Y = data['diagnosis'].map({'B':0,'M':1})
    class_names = ["Benign","Malignant"]
    feature_names = list(X.columns)
    selected_dataset = TabularClassification(dataset_name="Cancer",X=X,Y=Y,feature_names=feature_names,class_names=class_names)
elif selected_dataset_title == "titanic":
    #titanic_dataset
    data = pd.read_csv("./datasets/titanic.csv")
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
    selected_dataset = TabularClassification(dataset_name="Titanic",X=data[features],Y=data.Survived,feature_names=features,class_names=class_names,categorical_features=categorical_features)

elif selected_dataset_title == "wine":
    # wine_dataset
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    selected_dataset = TabularClassification(dataset_name="Wine",X=df,Y=wine_data.target,feature_names=wine_data.feature_names,class_names=list(wine_data.target_names))

elif selected_dataset_title == "iris":
    #iris_flower
    data = pd.read_csv("./datasets/iris.csv")
    iris_type_dict = {"variety" : {"Setosa" : 0, "Versicolor" : 1,"Virginica" : 2}}
    data = data.replace(iris_type_dict)
    iris_class_names = ["Setosa","Versicolor","Virginica"]
    iris_features = list(data.columns)
    prediction_feature = "variety"
    iris_features.remove(prediction_feature)
    selected_dataset = TabularClassification(dataset_name="Iris",X=data[iris_features],Y=data[prediction_feature],feature_names=iris_features,class_names=iris_class_names)

elif selected_dataset_title == "loan":
    #loan
    data = pd.read_csv("./datasets/loan.csv")
    Y = data['Loan_Status'].map({"N":0,"Y":1})
    data.drop(["Loan_Status","Loan_ID"],axis=1,inplace=True)

    for col in data.columns:
        if data[col].isnull().sum() != 0:
            data[col].fillna(data[col].value_counts().idxmax(),inplace = True)

    loan_features = list(data.columns)
    loan_categorical_features = [col for col in data.columns if data[col].dtype == "object"]
    loan_class_names = ["Loan_Denied","Loan_Approved"]
    selected_dataset = TabularClassification(dataset_name="Loan",X=data,Y=Y,feature_names=loan_features,class_names=loan_class_names,categorical_features=loan_categorical_features)

elif selected_dataset_title == "diabetes":
    #Diabetes_dataset
    data = pd.read_csv("./datasets/diabetes.csv")
    Y = data["Outcome"]
    diabetes_features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    diabetes_class_names = ["Negative","Positive"]
    selected_dataset = TabularClassification(dataset_name="Diabetes",X=data[diabetes_features],Y=Y,feature_names=diabetes_features,class_names=diabetes_class_names)

elif selected_dataset_title == "reddit":
    #reddit_dataset
    data = pd.read_csv("./datasets/reddit.csv")
    reddit_type_dict = {"Topic" : {"Physics" : 0, "Chemistry" : 1,"Biology" : 2}}
    data = data.replace(reddit_type_dict)
    reddit_class_names = ["Physics","Chemistry","Biology"]
    X = data['Comment'].values
    Y = data['Topic'].values
    selected_dataset = TextClassification(dataset_name="Reddit",X=X,Y=Y,class_names=reddit_class_names)


if args.explain and args.evaluation:
    sys.exit("Invalid command")

if not args.explain:
    #benchmarking
    if not args.evaluation or args.evaluation == "fidelity" and selected_dataset_title in Tabular:
        fidelity_scores = selected_dataset.calculate_explainers_fidelity()
        print(f"Fidelity scores of the explainers over the {selected_dataset_title} dataset")
        print(fidelity_scores)
    if not args.evaluation or args.evaluation == "runtime":
        print(f"Runtime of the explainers over the {selected_dataset_title} dataset")
        runtime = selected_dataset.get_average_explanation_times()
        print(runtime)
else:
    #explaining an instance
    selected_dataset.get_explanations(index_to_explain=args.explain,output=True)








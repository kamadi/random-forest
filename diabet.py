import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HEADERS = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

dataset = pd.read_csv("res/diabetes.csv")

train_x, test_x, train_y, test_y = train_test_split(dataset[HEADERS[1:-1]], dataset[HEADERS[-1]])

clf = RandomForestClassifier()

clf.fit(train_x, train_y)

print("Training Accuracy  :: ", clf.score(train_x,train_y))

print("Test Accuracy  :: ", clf.score(test_x,test_y))
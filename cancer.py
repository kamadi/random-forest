import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

dataset = pd.read_csv("res/breast-cancer-wisconsin.csv")

dataset = dataset[dataset[HEADERS[6]] != '?']

print(HEADERS[-1])

train_x, test_x, train_y, test_y = train_test_split(dataset[HEADERS[1:-1]], dataset[HEADERS[-1]],train_size=0.7)

clf = RandomForestClassifier()

clf.fit(train_x, train_y)

print("Training Accuracy  :: ", clf.score(train_x,train_y))

print("Test Accuracy  :: ", clf.score(test_x,test_y))




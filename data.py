
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
heart_data = pd.read_csv("heart_data.csv")


def trainingModel(userInfo):
    X = heart_data.drop("target", axis=1)
    y = heart_data["target"]
    clf = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)

    # Reshape the data to fit the training model
    reshapeData = pd.Series(userInfo).values.reshape(1, -1)

    # Use the SKLearn prediction function
    prediction = clf.predict(reshapeData)
    print(prediction)

    if(prediction == 0):
        print("Low chance of heart disease")
    if(prediction == 1):
        print("High chance of heart disease")

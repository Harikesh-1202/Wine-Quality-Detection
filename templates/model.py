import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
wine_dataset = pd.read_csv('winequality-red.csv')
correlation = wine_dataset.corr()
X=wine_dataset.drop('quality',axis=1)
Y=wine_dataset['quality']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
model = RandomForestClassifier()
model.fit(X_train,Y_train)
#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy :',test_data_accuracy)
with open("wine_quality_model.pkl", "wb") as f:
  pickle.dump(model, f)
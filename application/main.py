import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv(f"{str(Path.home())}\\Documents\\diabetes_ML\\dataset\\diabetes\\diabetes.csv")
colunas = df.columns
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


svm_model = SVC()
svm_model.fit(X_train, y_train)
y_predict = svm_model.predict(X_test)
print(classification_report(y_test, y_predict))

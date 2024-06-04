# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:27:41 2024
@author: caidara01
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Lire le fichier dans un DataFrame
data = pd.read_csv("billets.csv", sep = ";")

# Prepare les données
data["is_genuine"].replace([True, False], [1, 0], inplace=True)

# stocker les Nan dans df_na
df_na = data[data["margin_low"].isnull()]
df = data.dropna()

X = df[["margin_up", "is_genuine"]]
y = df["margin_low"]

linear_model = LinearRegression()
linear_model.fit(X, y)

# Prediction des valeures Nan
predicted_na = linear_model.predict(df_na[["margin_up", "is_genuine"]])
data.loc[data["margin_low"].isnull(), "margin_low"] = predicted_na

# definir les features et le target
features = data[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
target = data['is_genuine']

# Mise en Echelle des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Appliquer les K-Means()
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Ajouter les clusters
features['cluster'] = clusters

# Separer le modele en en train et test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creez le modele et l'entrainer avec la Logistic Regression model
logistic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression())
])

logistic_model.fit(X_train, y_train)

# predictions
y_pred = logistic_model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Enregistrez le modéle
joblib.dump(linear_model, 'linear_model.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
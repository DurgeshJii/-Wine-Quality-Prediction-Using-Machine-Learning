# Wine Quality Prediction Using Machine Learning
# Project Day - 24

# -------------------- Import Dependencies --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------- Data Collection --------------------
wine_dataset = pd.read_csv("winequality.csv")

# Preview dataset
print(wine_dataset.head())
print("Dataset Shape:", wine_dataset.shape)

# -------------------- Missing Values --------------------
print("\nMissing Values:")
print(wine_dataset.isnull().sum())

# -------------------- Data Analysis --------------------
print("\nStatistical Summary:")
print(wine_dataset.describe())

# -------------------- Data Visualization --------------------
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.title("Wine Quality Count")
plt.show()

# Volatile acidity vs Quality
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.title("Volatile Acidity vs Quality")
plt.show()

# Citric acid vs Quality
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
plt.title("Citric Acid vs Quality")
plt.show()

# -------------------- Correlation Heatmap --------------------
correlation = wine_dataset.corr()

plt.figure(figsize=(10,10))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt='.1f',
    annot=True,
    annot_kws={'size':8},
    cmap='Blues'
)
plt.title("Correlation Heatmap")
plt.show()

# -------------------- Data Preprocessing --------------------
X = wine_dataset.drop('quality', axis=1)

# Label Binarization
Y = wine_dataset['quality'].apply(lambda y: 1 if y >= 7 else 0)

print("\nFeature Data Shape:", X.shape)
print("Label Data Shape:", Y.shape)

# -------------------- Train Test Split --------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

# -------------------- Model Training --------------------
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# -------------------- Model Evaluation --------------------
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("\nModel Accuracy:", test_data_accuracy)

# -------------------- Predictive System --------------------
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)

input_df = pd.DataFrame([input_data], columns=feature_names)

prediction = model.predict(input_df)

if prediction[0] == 1:
    print("Good Quality Wine 🍷")
else:
    print("Bad Quality Wine ❌")

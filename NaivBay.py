import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# Step 1: Load the dataset
data = pd.read_csv(r"C:\DATA SCIENCE\DATASETS\water_potability.csv")

#Checking null values
# print(data.isnull().sum())

# Fill null values
ph_mean=data['ph'].mean()
data['ph'].fillna(ph_mean, inplace=True)

Sulfate_mean=data['Sulfate'].mean()
data['Sulfate'].fillna(Sulfate_mean, inplace=True)

Trihalomethanes_mean=data['Trihalomethanes'].mean()
data['Trihalomethanes'].fillna(Trihalomethanes_mean, inplace=True)

# print(data.isnull().sum())

# Step 2: Data Preprocessing
# Splitting data into features (X) and target variable (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
X_train= st_x.fit_transform(X_train)
X_test= st_x.transform(X_test)

# Step 3: Model Building
# Initializing and training the Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()
# Fit the model to the training data
naive_bayes.fit(X_train, y_train)

# Making predictions on the test set
y_pred = naive_bayes.predict(X_test)

# Step 4: Model Evaluation
# Evaluating the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from Adaline import AdalineGD as Adaline
import matplotlib.pyplot as plt

 

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Convert the binary target variable to match the Adaline output (1 for benign, 0 for malignant)
# Inverting the labels because originally 0 is malignant and 1 is benign
y = np.where(y == 0, 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

X_train_std.shape, X_test_std.shape, y_train.shape, y_test.shape

#Create and train the Adaline model
adaline = Adaline(n_iter=100, eta=0.01, random_state=1)
adaline.fit(X_train_std, y_train)

#Visualise the Adaline errors over Epochs
plt.figure(num="Adaline Errors over Epochs", figsize=(20,10))
plt.plot(range(1, len(adaline.losses_) + 1), adaline.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

#Evaluate the model 
from sklearn.metrics import accuracy_score

#Predictions
y_pred = adaline.predict(X_test_std)

#Accuracy 
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
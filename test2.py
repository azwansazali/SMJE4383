import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize and train SVM classifier
clf = svm.SVC()
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)

# Define colors for the plot
colors = np.array(['red', 'green', 'blue'])

# Plotting the data
plt.figure(figsize=(10, 6))

# Scatter plot for each predicted class
plt.scatter(X[:, 0], X[:, 1], c=colors[predictions], edgecolor='k', s=50, label='Predicted Class')

# Adding labels and title
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.title('SVM Classifier Predictions on Iris Dataset')
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.show()

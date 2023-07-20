import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# Dataset Creation
X, y = make_blobs(n_samples=300, n_features=2, cluster_std=.6
                  ,centers= [(1,4.5), (2,0.5)])
#plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired);

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)

# SVM Classifier
svm_clf = SVC(kernel="linear", C=1000)
svm_clf.fit(X, y)

# Model Performance
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
# Plot hyperplane
DecisionBoundaryDisplay.from_estimator(
    svm_clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# Plot support vectors
ax.scatter(
    svm_clf.support_vectors_[:, 0],
    svm_clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()

y_pred = svm_clf.predict(X_test)
print('Accuracy: {}%'.format (accuracy_score(y_test, y_pred)*100))


# Iris Dataset
iris_df = pd.read_csv("iris.csv")
iris_df[iris_df.columns[-1]] = iris_df[iris_df.columns[-1]].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0,1,2])
X, y = iris_df.iloc[:,[2,3]].to_numpy(), iris_df.iloc[:,-1:].to_numpy()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)

# SVM Classifier
svm_clf = SVC(kernel="linear", C=1000)
svm_clf.fit(X, y)

# Model Performance
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
# Plot hyperplane
DecisionBoundaryDisplay.from_estimator(
    svm_clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# Plot support vectors
ax.scatter(
    svm_clf.support_vectors_[:, 0],
    svm_clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()

y_pred = svm_clf.predict(X_test)
print('Accuracy: {}%'.format (accuracy_score(y_test, y_pred)*100))
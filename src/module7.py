# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics, datasets
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
fruits = pd.read_table(
    'D:\\projects\\python_courses\\datasets\\ml\\fruit_data_with_colors.txt')
fruits.head()
# %%
fruits['fruit_name'].unique()
# %%
labels = fruits[['fruit_label','fruit_name','fruit_subtype']]
labels.head()

# %%
X = fruits[['mass','width','height','color_score']]
y = fruits[['fruit_label']]
# %%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
# %%
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(log_reg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(log_reg.score(X_test, y_test)))

#%%

clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))
#%%

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(X_test, y_test)))



# %%
digits = datasets.load_digits()

#%%
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# %%
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# %%
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# %%
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#%%
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

# %%


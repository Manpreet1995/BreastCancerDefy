from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
# dict_keys(['DESCR', 'feature_names', 'target_names', 'target', 'data'])
# Print full description by running:
print(cancer['DESCR'])
# 569 data points with 30 features
cancer['data'].shape
X = cancer['data']
y = cancer['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
#StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
len(mlp.coefs_)
len(mlp.coefs_[0])
len(mlp.intercepts_[0])
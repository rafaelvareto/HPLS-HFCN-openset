from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'nu': [0.001,0.5], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
svr = svm.OneClassSVM()
clf = GridSearchCV(svr, parameters)
clf.fit(iris.data[1:90], iris.target[1:90])
sorted(clf.cv_results_.keys())
#!/usr/bin/env python
# coding=utf-8
from sklearn import svm
from sklearn.externals import joblib
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
# 训练
clf.fit(X, y)
# 计算结果
result = clf.predict([[0.2, 0.1]])
print result

# 保存模型
joblib.dump(clf, 'svm.model')

# 下次使用加载模型 只导入相关的包即可:from sklearn.externals import joblib
# 别的任何包和变量都可以不用
clf = joblib.load('svm.model')
# 直接使用
print clf.predict([[2.3, 1.8]])

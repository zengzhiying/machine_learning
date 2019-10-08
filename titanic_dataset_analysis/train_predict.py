#!/usr/bin/env python3.6
# coding=utf-8

import pandas
import numpy
from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    train_dataset = pandas.read_csv('train.csv')
    test_dataset = pandas.read_csv('test.csv')
    test_dataframe = pandas.read_csv('gender_submission.csv')
    test_labels = test_dataframe.Survived.to_numpy()

    train_nrow = train_dataset.shape[0]
    test_nrow = test_dataset.shape[0]

    # PassengerId: 乘客id号; Survived: 标签 是否是幸存者, 1: 存活, 0: 死亡
    # Pclass: 船舱等级, 数字越小等级越高 1 2 3; Name: 乘客姓名
    # Sex: 乘客性别, male: 男, female: 女; Age: 乘客年龄; SibSp: 兄弟姐妹人数 [统计: 0,1,2,3,4,5,8 共7种情况]
    # Parch: 父母和小孩人数 [0,1,2,3,4,5,6,9 共8种情况]; Ticket: 票编号; Fare: 票价格
    # Cabin: 船舱号; Embarked: 起始点, 即登船口 [S, C, Q 3种情况]
    print(train_dataset.columns)
    print(train_dataset.Pclass.value_counts())
    print(train_dataset.SibSp.value_counts())
    print(train_dataset.Parch.value_counts())

    # 查看缺失值情况
    # 训练集 Age, Cabin, Embarked有缺失
    train_dataset.isnull().sum()
    # 测试集 Age, Cabin, Fare 存在缺失
    test_dataset.isnull().sum()

    # 训练和预测用到的feature: Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked
    # 年龄用均值填充
    train_dataset['Age'].fillna(train_dataset.Age.mean(), inplace=True)
    test_dataset['Age'].fillna(test_dataset.Age.mean(), inplace=True)
    # 票价填充
    test_dataset['Fare'].fillna(test_dataset.Fare.mean(), inplace=True)
    # 将Cabin设置为1和0, 存在船舱号为1, 缺失为0
    train_dataset.loc[train_dataset.Cabin.notnull(), 'Cabin'] = 1
    train_dataset.loc[train_dataset.Cabin.isnull(), 'Cabin'] = 0
    test_dataset.loc[test_dataset.Cabin.notnull(), 'Cabin'] = 1
    test_dataset.loc[test_dataset.Cabin.isnull(), 'Cabin'] = 0
    # Embarked使用数量较多的代替, 即'S'
    train_dataset['Embarked'].fillna('S', inplace=True)

    # 整理数据集输入: Pclass, Sex, SibSp, Parch, Cabin, Embarked 采用one-hot编码
    # Age, Fare 采用标准化输入
    # 输入feature顺序: Pclass Sex SibSp Parch Cabin Embarked Age Fare
    train_Pclass = pandas.get_dummies(train_dataset.Pclass, prefix='Pclass')
    test_Pclass = pandas.get_dummies(test_dataset.Pclass, prefix='Pclass')

    train_Sex = pandas.get_dummies(train_dataset.Sex, prefix='Sex')
    test_Sex = pandas.get_dummies(test_dataset.Sex, prefix='Sex')

    train_SibSp = pandas.get_dummies(train_dataset.SibSp, prefix='SibSp')
    test_SibSp = pandas.get_dummies(test_dataset.SibSp, prefix='SibSp')

    dataset_Parch = pandas.get_dummies(pandas.concat([train_dataset.Parch, test_dataset.Parch], axis=0), prefix='Parch')
    train_Parch = dataset_Parch[:train_nrow]
    test_Parch = dataset_Parch[train_nrow:]
    # test_Parch = pandas.get_dummies(test_dataset.Parch, prefix='Parch')

    train_Cabin = pandas.get_dummies(train_dataset.Cabin, prefix='Cabin')
    test_Cabin = pandas.get_dummies(test_dataset.Cabin, prefix='Cabin')

    train_Embarked = pandas.get_dummies(train_dataset.Embarked, prefix='Embarked')
    test_Embarked = pandas.get_dummies(test_dataset.Embarked, prefix='Embarked')

    # Age, Fare对全部样本计算均值和方差
    # dataset_ages = numpy.concatenate((train_dataset.Age.to_numpy(), test_dataset.Age.to_numpy()), axis=0)
    age_mean = pandas.concat([train_dataset.Age, test_dataset.Age], axis=0).mean()
    age_std = pandas.concat([train_dataset.Age, test_dataset.Age], axis=0).std()
    fare_mean = pandas.concat([train_dataset.Fare, test_dataset.Fare], axis=0).mean()
    fare_std = pandas.concat([train_dataset.Fare, test_dataset.Fare], axis=0).std()

    age_scale_train = (train_dataset.Age - age_mean) / age_std
    age_scale_test = (test_dataset.Age - age_mean) / age_std

    fare_scale_train = (train_dataset.Fare - fare_mean) / fare_std
    fare_scale_test = (test_dataset.Fare - fare_mean) / fare_std

    train_inputs = pandas.concat([train_Pclass, train_Sex, train_SibSp, train_Parch,
                                  train_Cabin, train_Embarked, age_scale_train, fare_scale_train], axis=1)
    test_inputs = pandas.concat([test_Pclass, test_Sex, test_SibSp, test_Parch,
                                 test_Cabin, test_Embarked, age_scale_test, fare_scale_test], axis=1)
    train_inputs = train_inputs.to_numpy()
    test_inputs = test_inputs.to_numpy()

    print(train_inputs.shape, test_inputs.shape)

    train_labels = train_dataset.Survived.to_numpy()

    # with open('gender_submission.csv') as f:
    #     test_content = f.read().strip()
    # test_labels = [int(r.split(',')[1]) for r in test_content.split(' ')]
    # test_labels = numpy.array(test_labels)

    clf = svm.SVC(gamma='scale', kernel='rbf')
    clf.fit(train_inputs, train_labels)

    predicts = clf.predict(test_inputs)
    # print(predicts)
    # print(predicts == test_labels)
    # acc: 93.54%
    print("acc1: %f" % (numpy.where(predicts == test_labels)[0].size / test_labels.shape[0]))

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_inputs, train_labels)
    predicts = lr.predict(test_inputs)
    # print(predicts)
    # print(predicts == test_labels)
    # 置信度 n*2 分别表示分类0和1的置信度
    # print(lr.predict_proba(test_inputs))
    # acc: 93.30%
    print("acc2: %f" % (numpy.where(predicts == test_labels)[0].size / test_labels.shape[0]))
    print("acc2: %f" % lr.score(test_inputs, test_labels))

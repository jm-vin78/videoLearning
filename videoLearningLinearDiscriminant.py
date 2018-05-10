from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
import pandas
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

colnames = ['idVideo', 'numberOfViews', 'likes', 'dislikes', 'mistakes', 'presentation', 'informative', 'quality', 'numberOfSubscribers', 'averageViewsAllVideos',
            'likesToDislikes', 'likesDislikesDifference', 'viewsRatio']
data = pandas.read_csv('C:/Users/yulia/videodataWork.csv', usecols=colnames)
surveyData = data[data['mistakes'].notnull()]
surveyData.to_csv('C:/Users/yulia/surveyData.csv', sep=',')

trainingSet = surveyData.tail(n=80)
testSet = surveyData.head(n=20)

# choose columns for training set
featureColumns = ['numberOfViews', 'likes', 'dislikes', 'numberOfSubscribers', 'averageViewsAllVideos', 'likesToDislikes', 'likesDislikesDifference', 'viewsRatio']
X = trainingSet.loc[:, featureColumns]
print(X.shape)

tests = ['mistakes', 'informative', 'presentation', 'quality']
for predData in tests:
    y = trainingSet.loc[:, predData]
    print(y.shape)

    # clf = SVC()
    # print(clf.fit(X, y))

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)


    testSet = testSet.drop(predData, axis=1)
    XNew = testSet.loc[:, featureColumns]
    # print(XNew)


    newPredClass = clf.predict(XNew)


    surveyDataMistakes = list(surveyData[predData].astype(int))
    tenFirst = surveyDataMistakes[:20]

    print(tenFirst)
    print(newPredClass)

    print(mean_squared_error(newPredClass, tenFirst))

    print(len([i for i, j in zip(newPredClass, tenFirst) if i == j]))
    print(len(testSet))
    print(len([i for i, j in zip(newPredClass, tenFirst) if i == j])/len(testSet))
    print("\n")

    # plt.scatter(tenFirst, newPredClass)
    # plt.xlabel("True Values")
    # plt.ylabel("Predictions")
    #
    # plt.show()


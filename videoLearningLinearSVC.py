import pandas
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

colnames = ['idVideo', 'numberOfViews', 'likes', 'dislikes', 'mistakes', 'presentation', 'informative', 'quality', 'numberOfSubscribers', 'averageViewsAllVideos',
            'likesToDislikes', 'likesDislikesDifference', 'viewsRatio']
data = pandas.read_csv('C:/Users/yulia/videodataWork.csv', usecols=colnames)
surveyData = data[data['mistakes'].notnull()]
surveyData.to_csv('C:/Users/yulia/surveyData.csv', sep=',')
#surveyData.fillna(0)
print(np.any(np.isnan(surveyData)))
print(np.all(np.isfinite(surveyData)))

# print(len(surveyData))

trainingSet = surveyData.tail(n=80)
testSet = surveyData.head(n=20)


featureColumns = ['numberOfViews', 'likes', 'dislikes', 'numberOfSubscribers', 'averageViewsAllVideos', 'likesToDislikes', 'likesDislikesDifference', 'viewsRatio']
X = trainingSet.loc[:, featureColumns]
print(X.shape)

y = trainingSet.mistakes
print(y.shape)

# clf = SVC()
# print(clf.fit(X, y))

clf = LinearSVC(random_state=0)
clf.fit(X, y)

print(clf.predict([[106, 3, 3, 43, 35.87, 0.57, 0, 2.96]]))

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)

#print(neigh.predict([[106, 3, 3, 43, 35.87, 0.57, 0, 2.96]]))

testSet = testSet.drop('mistakes', axis=1)
XNew = testSet.loc[:, featureColumns]
# print(XNew)

# newPredClass = neigh.predict(XNew)

newPredClass = clf.predict(XNew)
print(newPredClass)

surveyDataMistakes = list(surveyData['mistakes'].astype(int))
tenFirst = surveyDataMistakes[:20]

print(tenFirst)
print(mean_squared_error(newPredClass, tenFirst))



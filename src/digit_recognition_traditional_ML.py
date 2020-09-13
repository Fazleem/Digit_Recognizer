import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.io as sci
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from pathlib import Path

NumberData = Path(__file__).resolve().parent.parent / "data/NumberRecognition.mat"
dataFrame = sci.loadmat(str(NumberData))
"""extract training data  and reshape to get a new shape with out changing the data - 2D (750,784)"""
trainingData8 = dataFrame["imageArrayTraining8"]
trainingData9 = dataFrame["imageArrayTraining9"]
trainingData8 = trainingData8.transpose()
trainingData9 = trainingData9.transpose()
reshapedTrainingData8 = trainingData8.reshape(750, (28 * 28))
reshapedTrainingData9 = trainingData9.reshape(750, (28 * 28))
concatTraining = np.concatenate((reshapedTrainingData8, reshapedTrainingData9))
print(concatTraining.shape)

"""label training data"""
eightLabel = np.ones(750) * 8
nineLabel = np.ones(750) * 9
concatLabels = np.concatenate((eightLabel, nineLabel))

"""extract testing data and do the same as training data"""
testingData8 = dataFrame["imageArrayTesting8"]
testingData9 = dataFrame["imageArrayTesting9"]
testingData8 = testingData8.transpose()
testingData9 = testingData9.transpose()
reshapedTestingData8 = testingData8.reshape(250, (28 * 28))
reshapedTestingData9 = testingData9.reshape(250, (28 * 28))
concatTestingData = np.concatenate((reshapedTestingData8, reshapedTestingData9))
"""label testing data"""
TestingEightLabel = np.ones(250) * 8
TestingNineLabel = np.ones(250) * 9
concatTestLabels = np.concatenate((TestingEightLabel, TestingNineLabel))


"""pass variables to train and test model using KNN"""
X = []
Y = []

for k in range(1, 21):
    X.append(k)
    classifierN = KNeighborsClassifier(k, n_jobs=-1)
    classifierN.fit(concatTraining, concatLabels)
    # pass testing data and calculate the accuracy using the score function

    # model_score = classifierN.score(concatTestingData,concatTestLabels)
    predicted_labels = classifierN.predict(concatTestingData)
    accuracy = metrics.accuracy_score(concatTestLabels, predicted_labels)
    errorRate = 1 - accuracy
    Y.append(errorRate)
    print("k value:", k, "error rate is:", errorRate)


# Plot the error rate
sbn.set(style="darkgrid")
plt.title("Describes error rate with K value")
plt.xlabel("K value")
plt.ylabel("Error Rate")
plt.plot(X, Y, lw=2)
plt.show()

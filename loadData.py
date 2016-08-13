import numpy
import math

dataDir = "data-2016-08-10"
minibatchSize = 200 # I guessed. No idea what is optimal.

numpy.random.seed(19900515)

dataDir = "data/" + dataDir

trainDataIn = numpy.loadtxt(dataDir + "/numerai_training_data.csv", skiprows=1, delimiter=',')
numpy.random.shuffle(trainDataIn)

splits = numpy.split(trainDataIn, [math.floor(trainDataIn.shape[0]*0.9)])
trainData = splits[0]
testData = splits[1]

minibatchIdx = 0

def getFeatures(mat):
    return mat[:, :-1]

def getYs(mat):
    return mat[:, -1:]

# This actually hurts performance.
# means_train = numpy.mean(getFeatures(trainData), axis=0)
# centered_train = getFeatures(trainData) - means_train
# cov_mat_train = numpy.cov(centered_train, rowvar=False)
# U,S,V = numpy.linalg.svd(cov_mat_train)
# centered_all = getFeatures(trainDataIn) - means_train
# rot_all = numpy.dot(centered_all, U)
# white_all = rot_all / numpy.sqrt(S + 1)
# back_again_all = numpy.dot(white_all * numpy.sqrt(S + 1e-5), numpy.linalg.inv(U)) + means_train
# trainDataIn_orig = numpy.copy(trainDataIn)
# trainDataIn[:, :-1] = white_all

def getMinibatch():
    global minibatchIdx
    if minibatchIdx + minibatchSize < trainData.shape[0]:
        minibatch = trainData[minibatchIdx:minibatchIdx+minibatchSize, :]
        minibatchIdx += minibatchSize
    else:
        minibatch = trainData[minibatchIdx:, :]
        minibatchIdx = 0
        numpy.random.shuffle(trainData)
    return getFeatures(minibatch), getYs(minibatch)

def getTrainFeatures():
    return getFeatures(trainData)

def getTrainYs():
    return getYs(trainData)

def getTestFeatures():
    return getFeatures(testData)

def getTestYs():
    return getYs(testData)

tournamentDataIn = numpy.loadtxt(dataDir + "/numerai_tournament_data.csv", skiprows=1, delimiter=',')

def getTournamentData():
    return tournamentDataIn[:, 1:]

def getTournamentTids():
    return tournamentDataIn[:, 0:1]

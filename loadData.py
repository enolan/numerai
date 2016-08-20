import numpy
import math

dataDir = "data-2016-08-17"

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

means_train = numpy.mean(getFeatures(trainData), axis=0)
centered_all = getFeatures(trainDataIn) - means_train
trainDataIn_orig = numpy.copy(trainDataIn)
trainDataIn[:, :-1] = centered_all

def getMinibatch(minibatch_size):
    global minibatchIdx
    if minibatchIdx + minibatch_size < trainData.shape[0]:
        minibatch = trainData[minibatchIdx:minibatchIdx+minibatch_size, :]
        minibatchIdx += minibatch_size
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
tournamentDataIn_orig = numpy.copy(tournamentDataIn)
centered_tournament = tournamentDataIn[:, 1:] - means_train
tournamentDataIn[:, 1:] = centered_tournament

def getTournamentData():
    return tournamentDataIn[:, 1:]

def getTournamentTids():
    return tournamentDataIn[:, 0:1]

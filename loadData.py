import numpy
import math

dataDir = "data-2016-08-03"
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

def getMinibatch():
    global minibatchIdx
    if minibatchIdx + minibatchSize < trainData.shape[0]:
        minibatch = trainData[minibatchIdx:minibatchIdx+minibatchSize, :]
        minibatchIdx += minibatchSize
    else:
        minibatch = trainData[minibatchIdx:, :]
        minibatchIdx = 0
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

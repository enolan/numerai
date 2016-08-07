import numpy
import math

numpy.random.seed(19900515)

dataDir = "data-2016-08-03"
dataDir = "data/" + dataDir

trainDataIn = numpy.loadtxt(dataDir + "/numerai_training_data.csv", skiprows=1, delimiter=',')
numpy.random.shuffle(trainDataIn)

splits = numpy.split(trainDataIn, [math.floor(trainDataIn.shape[0]*0.9)])
trainData = splits[0]
testData = splits[1]

def getFeatures(mat):
    return mat[:, :-1]

def getYs(mat):
    return mat[:, -1:]

def getTrainFeatures():
    return getFeatures(trainData)

def getTrainYs():
    return getYs(trainData)

def getTestFeatures():
    return getFeatures(testData)

def getTestYs():
    return getYs(testData)

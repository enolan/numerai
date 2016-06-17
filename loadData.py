import numpy

dataDir = "data-2016-06-17"

dataDir = "data/" + dataDir

train = numpy.loadtxt(dataDir + "/numerai_training_data.csv", skiprows=1, delimiter=',')

ysData = train[:, 21]

def getTrainFeatures():
    return train[:, 0:20]

def getTrainYs():
    return train[:, 21]

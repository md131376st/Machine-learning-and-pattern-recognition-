import sklearn.datasets
import numpy as np
import pylab
import scipy


class LDA:
    def __init__(self, m):
        self.m = m
        self.data = np.ndarray
        self.labels = np.ndarray
        self.SampleSizeDiffClass = np.ndarray
        self.eachClassMean = np.ndarray
        self.sb = 0
        self.sw = 0
        self.LoadData()
        self.NumberOfClassSample()
        self.SumAllDataValues = sum(self.SampleSizeDiffClass)
        self.SampleMean = self.CalculateMean(self.data)
        self.CalculateClassMean()
        self.CalculateSb()
        self.CalculateSW()
        self.CalculeteEginValue()
        self.PlotFunction()

        pass

    def CalculeteEginValue(self):
        s, U = scipy.linalg.eigh(self.sb, self.sw)
        W = U[:, ::-1][:, 0:self.m]
        UW, _, _ = np.linalg.svd(W)
        self.eginVector = UW[:, 0:self.m]

    def PlotFunction(self):
        projection_list = np.dot(self.eginVector.T, self.data)
        men = projection_list[:, self.labels == 0]
        women = projection_list[:, self.labels == 1]

        pylab.scatter(men[0, :], men[1, :])
        pylab.scatter(women[0, :], women[1, :])
        pylab.xlabel('Principal component 1')
        pylab.ylabel('Principal component 2')
        pylab.legend(['men', 'women', "allData"])
        pylab.show()

    def NumberOfClassSample(self):
        self.SampleSizeDiffClass = np.array([np.sum(self.labels == i) for i in set(self.labels)])

    def CalculateClassMean(self):
        self.eachClassMean = np.array([self.CalculateMean(self.data[:, self.labels == i]) for i in set(self.labels)]).T

    def CalculateDiffMeanClassAndMeanDataset(self):
        return self.eachClassMean - self.VectorCol(self.SampleMean)

    def CalculateSb(self):
        for i in range(self.SampleSizeDiffClass.size):
            # mc-m
            diff_means = self.CalculateDiffMeanClassAndMeanDataset()[:, i:i + 1]
            self.sb += self.SampleSizeDiffClass[i] * np.dot(diff_means, diff_means.T)
        self.sb /= self.SumAllDataValues

    def CalculateSW(self):
        for i in range(self.SampleSizeDiffClass.size):
            classData = self.data[:, self.labels == i]
            centerData = classData - self.VectorCol(self.eachClassMean.T[i])
            SWc = 1 / self.SampleSizeDiffClass[i] * np.dot(centerData, centerData.T)
            self.sw += self.SampleSizeDiffClass[i] * SWc
        self.sw /= self.SumAllDataValues

    def LoadData(self):
        self.data = np.genfromtxt("Train.txt", delimiter=",")
        self.labels = self.data[:, -1].T
        self.data = self.data[:, :-1].T

    def VectorCol(self, data):
        return data.reshape((data.size, 1))

    def CalculateMean(self, data):
        return data.mean(1)

lda = LDA(5)
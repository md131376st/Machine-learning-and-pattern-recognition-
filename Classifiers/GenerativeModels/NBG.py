import numpy as np
from Info import Info
from Classifiers.GenerativeModels.MGC import MGC


# calculate Naive Bayes Gaussian Classifier
class NBG:
    def __init__(self):
        self.info = Info()
        self.classTypes = len(set(self.info.testlable))

        self.mgc = MGC()
        self.cov_classes_nbayes = []
        # calculate diagonal covariance class
        for i in range(self.classTypes):
            self.cov_classes_nbayes = self.mgc.cov_classes.append(
                self.mgc.cov_classes[i] * np.identity(self.info.data.shape[0]))

    def applyNaiveBayesOnTest(self):
        self.mgc.cov_classes = self.cov_classes_nbayes
        self.mgc.applyMGCOnTest()

    def checkAcc(self):
        self.mgc.checkAcc()
        pass


NaiveBayes = NBG()
NaiveBayes.applyNaiveBayesOnTest()
NaiveBayes.checkAcc()

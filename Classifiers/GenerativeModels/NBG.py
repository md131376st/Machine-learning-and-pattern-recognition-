import numpy as np
from MGC import MGC


# calculate Naive Bayes Gaussian Classifier
class NBG:
    def __init__(self):
        self.mgc = MGC()
        self.cov_classes_nbayes = []
        # calculate diagonal covariance class
        for i in range(self.mgc.classTypes):
            self.cov_classes_nbayes.append(
                self.mgc.cov_classes[i] * np.identity(self.mgc.info.data.shape[0]))

    def applyNaiveBayesOnTest(self):
        self.mgc.cov_classes = self.cov_classes_nbayes
        self.mgc.applyMGCOnTest()

    def checkAcc(self):
        self.mgc.checkAcc()
        pass


if __name__ == "__main__":
    NaiveBayes = NBG()
    NaiveBayes.applyNaiveBayesOnTest()
    NaiveBayes.checkAcc()

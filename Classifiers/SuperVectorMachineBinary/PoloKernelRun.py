from Classifiers.SuperVectorMachineBinary.KSVM import KSVM
from Data.Info import KFold

if __name__ == "__main__":
    KFold = KFold(3)
    listeps = [0, 1]
    listc = [10 ** -3, 0.1, 1]
    listd = [2,1]
    listC = [0.1, 1, 2]
    for C in listc:
        for d in listd:
            for eps in listeps:
                for c in listC:
                    for i in range(KFold.k):
                        LinearSVM = KSVM(KFold.infoSet[i], 'Polynomial', c, 1, eps, C, d)
                        LinearSVM.applyTest()
                        KFold.addscoreList(LinearSVM.checkAcc())

                    KFold.ValidatClassfier('KernelSVM RBF  C=%.1f, K=1, eps=%f, c=%f,  d=%f ' % (
                        c, eps, C, d))
                    KFold.scoreList = []

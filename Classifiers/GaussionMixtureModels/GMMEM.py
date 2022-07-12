import numpy
import scipy.special
import hi as mymath


class GMMEM():
    pass


def GMM_EM(X, GMM, thresholdForEValues, model):
    llNew = None
    llOld = None
    G = len(GMM)
    N = X.shape[1]
    gmmnew = None
    while llOld is None or (llNew - llOld) > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = mymath.Log_pdf_MVG_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        gmmnew = GMM
        P = numpy.exp(SJ - SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mymath.make_row_matrix(gamma) * X).sum(1)
            S = numpy.dot(X, (mymath.make_row_matrix(gamma) * X).T)
            w = Z / N
            mu = mymath.make_col_matrix(F / Z)
            Sigma = S / Z - numpy.dot(mu, mu.T)
            # diag sigma
            if model == "diagnal":
                Sigma = Sigma * numpy.eye(Sigma.shape[0])
            # to apply threshold for EValues
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < thresholdForEValues] = thresholdForEValues
            SigmaNew = numpy.dot(U, mymath.make_col_matrix(s) * U.T)
            gmmNew.append((w, mu, SigmaNew))
        GMM = gmmNew
        print(llNew)
    print(llNew - llOld)
    return gmmnew


def GMM_SJoint(X, GMM):
    G = len(GMM)
    N = X.shape[1]
    SJ = numpy.zeros((G, N))
    for g in range(G):
        SJ[g, :] = mymath.Log_pdf_MVG_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
    return SJ


def GMM_ll_PerSample(X, GMM):
    G = len(GMM)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = mymath.Log_pdf_MVG_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
    return scipy.special.logsumexp(S, axis=0)


def gmmlbg(GMM, alpha):
    G = len(GMM)
    newGMM = []
    for g in range(G):
        (w, mu, CovarianMatrix) = GMM[g]
        U, s, _ = numpy.linalg.svd(CovarianMatrix)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        newGMM.append((w / 2, mu - d, CovarianMatrix))
        newGMM.append((w / 2, mu + d, CovarianMatrix))
    return newGMM


def GMM_EM_wrapper(DTR, LTR, DTE, LTE, thresholdForEValues, numberOfComponents, model="full"):
    mu = mymath.imperical_mean(DTR)
    C = mymath.Cov_matrix(mymath.normalize_data(DTR, mymath.imperical_mean(DTR)))
    gmm_init_0 = [(1.0, mu, C)]
    NewgmmEM = 0

    itteration = 1 + int(numpy.log2(numberOfComponents))

    mu_Cov_weight_pair_each_class = {}

    for label in set(list(LTR)):
        print("label=", label)
        gmm_init = gmm_init_0
        for i in range(itteration):
            print("GMM LEN", len(gmm_init))
            NewgmmEM = GMM_EM(DTR[:, LTR == label], gmm_init, thresholdForEValues, model)
            if i < itteration - 1:
                gmm_init = gmmlbg(NewgmmEM, 0.1)
        mu_Cov_weight_pair_each_class[label] = NewgmmEM

    final = numpy.zeros((len(set(LTE)), DTE.shape[1]))
    for i in set(list(LTR)):
        GMM = mu_Cov_weight_pair_each_class[i]
        SM = GMM_ll_PerSample(DTE, GMM)
        final[i] = SM

    density = numpy.exp(final)
    llr = numpy.log(density[1, :] / density[0, :])

    predictedLabelByGMM = final.argmax(0)
    error = (predictedLabelByGMM == LTE).sum() / LTE.size
    return predictedLabelByGMM, llr, error

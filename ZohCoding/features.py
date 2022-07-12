import numpy as np
import matplotlib.pyplot as plt


def LoadData():
    data = np.genfromtxt('./Train.txt', delimiter=",")
    return (data[:, -1].T, data[:, :-1].T)


L, D = LoadData()
features_Men = D[:, L == 0]
features_women = D[:, L == 1]
nameFeatures = ['Feature-1', 'Feature-2', 'Feature-3', 'Feature-4', 'Feature-5', 'Feature-6', 'Feature-7', 'Feature-8',
                'Feature-9', 'Feature-10', 'Feature-11', 'Feature-12']
plt.figure()
for f in range(12):
    plt.hist(features_Men[f], bins=20, density=True, alpha=0.4)
    plt.hist(features_women[f], bins=20, density=True, alpha=0.4)
    plt.legend(['men', 'women'])
    plt.xlabel(nameFeatures[f])
plt.show()

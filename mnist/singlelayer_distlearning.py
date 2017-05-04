import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'': lambda x: 'float: ' + str(x)})

nEpochs = 100000

def sample(X):
    U = np.zeros(X.shape)
    for i in range(len(X)):
        x = X[i]
        u = np.random.choice(len(x), p=x/sum(x))
        U[i][u] = 1
    return U

def hard_sample(X):
    U = np.zeros(X.shape)
    for i in range(len(X)):
        x = X[i]
        u = np.argmax(x)
        U[i][u] = 1
    return U

thresh = 0.1
X = (trX > thresh).astype("int8")
T = trY

nData = X.shape[0]

gain = 0.01

W1 = np.zeros((trX.shape[1],10)) + 0.5

log = []

batchSize = 512

for k in range(nEpochs):
    confusion = np.zeros((10,10))
    idxList = np.arange(nData)
    np.random.shuffle(idxList)

    acc_log = []
    hard_acc_log = []
    for j in range(nData/batchSize):
        idx = idxList[j*batchSize:(j+1)*batchSize]

        U0 = X[idx]

        H1 = U0.dot(W1)
        U1 = H1 / H1.sum(axis=1)[:,np.newaxis]
        hard_U1 = hard_sample(H1)

        Y = U1
        hard_Y = np.argmax(hard_U1,axis=1)

        for i in range(len(Y)):
            #print U0[i] * gain * U1[i][T[idx][i]]
            W1.T[T[idx][i]] += U0[i] * gain * U1[i][T[idx][i]] #/ U0[i].sum()
            confusion[T[idx][i]][hard_Y[i]] += 1

        #acc = ((Y == T[idx]).sum())/float(len(T[idx]))
        hard_acc = ((hard_Y == T[idx]).sum())/float(len(T[idx]))
        #acc_log.append(acc)
        hard_acc_log.append(hard_acc)
        #print W1

    print confusion
    #acc = sum(acc_log)/len(acc_log)
    hard_acc = sum(hard_acc_log)/len(hard_acc_log)
    print "-------- %s Epoch: HardAcc %s"%(k,hard_acc)

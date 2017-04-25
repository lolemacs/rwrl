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
        #u = np.argmax(x)
        u = np.random.choice(len(x), p=x/sum(x))
        U[i][u] = 1
    return U

nPoints = 1000
trX = trX[:nPoints]
trY = trY[:nPoints]

thresh = 0.1
X = (trX > thresh).astype("int8")
T = trY

nData = X.shape[0]

gain = 0.01

W1 = np.zeros((trX.shape[1],10)) + 1.0
#W1 = np.random.uniform(low=0.0001, high = 0.002, size = (trX.shape[1],10))

log = []

batchSize = 100

for k in range(nEpochs):
    confusion = np.zeros((10,10))
    #gain *= 0.999
    idxList = np.arange(nData)
    np.random.shuffle(idxList)

    batchlog = []
    for j in range(nData/batchSize):
        idx = idxList[j*batchSize:(j+1)*batchSize]

        U0 = X[idx]

        H1 = U0.dot(W1)
        #H1 = np.exp(H1)
        U1 = sample(H1)

        Y = np.argmax(U1,axis=1)

        for i in range(len(Y)):
            if Y[i] == T[idx][i]: W1.T[Y[i]] += U0[i] * gain / U0[i].sum()
            #if Y[i] != T[idx][i]: W1.T[Y[i]] -= U0[i] * gain #/ U0[i].sum()
            confusion[T[idx][i]][Y[i]] += 1

        acc = ((Y == T[idx]).sum())/float(len(T[idx]))
        batchlog.append(acc)

    print confusion
    acc = sum(batchlog)/len(batchlog)
    log.append(str(acc))
    print "%s Epoch: Acc %s"%(k,acc)
    #print W1
    #print U0
    #print H1[0]
    #print U1[0]
    #print T[idx][0]

f = open("dump","w")
f.write("\n".join(log))
f.close()

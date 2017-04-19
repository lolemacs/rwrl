import numpy as np

nEpochs = 2000

def sample(X):
    U = np.zeros(X.shape)
    for i in range(len(X)):
        x = X[i]
        u = np.random.choice(len(x), p=x/sum(x))
        U[i][u] = 1
    return U

def det_sample(X):
    U = np.zeros(X.shape)
    for i in range(len(X)):
        x = X[i]
        u = np.argmax(x)
        U[i][u] = 1
    return U

nData = 1000
X = np.eye(nData)
T = np.zeros(nData)
for i in range(0,len(T),2): T[i] = 1

nHidden1 = 20
nHidden2 = 20

gain = float(nHidden1)

W1 = np.zeros((nData,nHidden1))+1.0
W2 = np.zeros((nHidden1,nHidden2))+1.0
W3 = np.zeros((nHidden2,2))+1.0

log = []

batchSize = 9

for k in range(nEpochs):
    idxList = np.arange(nData)
    np.random.shuffle(idxList)

    batchlog = []
    for j in range(nData/batchSize):
        idx = idxList[j*batchSize:(j+1)*batchSize]
        U0 = X[idx]

        H1 = U0.dot(W1)
        U1 = sample(H1)

        H2 = U1.dot(W2)
        U2 = sample(H2)

        H3 = U2.dot(W3)
        U3 = sample(H3)

        Y = np.argmax(U3,axis=1)

        for i in range(len(Y)):
            if Y[i] == T[idx][i]:
                W1[np.argmax(U0[i])][np.argmax(U1[i])] += gain
                W2[np.argmax(U1[i])][np.argmax(U2[i])] += gain
                W3[np.argmax(U2[i])][np.argmax(U3[i])] += gain
        acc = ((Y == T[idx]).sum())/float(len(T[idx]))
        batchlog.append(acc)
    acc = sum(batchlog)/len(batchlog)
    log.append(str(acc))
    print "%s Epoch: Acc %s"%(k,acc)

H1 = X.dot(W1)
U1 = det_sample(H1)
H2 = U1.dot(W2)
U2 = det_sample(H2)
H3 = U2.dot(W3)
U3 = det_sample(H3)
Y = np.argmax(U3,axis=1)
acc = ((Y == T).sum())/float(len(T))
print acc

f = open("dump","w")
f.write("\n".join(log))
f.close()

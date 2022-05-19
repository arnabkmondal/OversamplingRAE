import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import os


SEED = 42
np.random.seed(SEED)

data_dir = '../imb_data'
os.makedirs(data_dir, exist_ok=True)

ds = 'cifar10'


if ds == 'mnist':
    (trainSetOri, trainLabOri), (testSetOri, testLabOri) = tf.keras.datasets.mnist.load_data()
    trainSetOri = np.expand_dims(trainSetOri, axis=-1)
    testSetOri = np.expand_dims(testSetOri, axis=-1)
    pointsInTrClass=((4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40))
    maxPTrClass, maxPTsClass, maxPTeClass = 4000, 800, 980
    pointsInTsClass=800
    pointsInTeClass= ((980, 500, 250, 187, 125, 87, 50, 25, 15, 10))
    IMG_SIZE = 28
    h, w, c = 28, 28, 1
    numClass = 10
elif ds == 'fashion':
    (trainSetOri, trainLabOri), (testSetOri, testLabOri) = tf.keras.datasets.fashion_mnist.load_data()
    trainSetOri = np.expand_dims(trainSetOri, axis=-1)
    testSetOri = np.expand_dims(testSetOri, axis=-1)
    pointsInTrClass=((4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40))
    maxPTrClass, maxPTsClass, maxPTeClass = 4000, 1000, 1000
    pointsInTsClass=800
    pointsInTeClass = ((1000, 500, 250, 187, 125, 87, 50, 25, 15, 10))
    IMG_SIZE = 28
    h, w, c = 28, 28, 1
    numClass = 10
elif ds == 'cifar10':
    (trainSetOri, trainLabOri), (testSetOri, testLabOri) = tf.keras.datasets.cifar10.load_data()
    trainLabOri = trainLabOri.flatten()
    testLabOri = testLabOri.flatten()
    pointsInTrClass=((4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80))
    maxPTrClass, maxPTsClass, maxPTeClass = 4500, 1000, 1000
    pointsInTsClass=800
    pointsInTeClass= ((1000, 500, 250, 187, 125, 87, 50, 25, 15, 10))
    IMG_SIZE = 32
    h, w, c = 32, 32, 3
    numClass = 10
elif ds == 'svhn':
    tfds_train, tfds_test = tfds.load('svhn_cropped', split=['train', 'test'], batch_size=-1, as_supervised=True)
    trainSetOri, trainLabOri = tfds.as_numpy(tfds_train)
    testSetOri, testLabOri = tfds.as_numpy(tfds_test)
    trainLabOri = np.array(trainLabOri).flatten()
    testLabOri = np.array(testLabOri).flatten()
    pointsInTrClass=((4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80))
    maxPTrClass, maxPTsClass, maxPTeClass = 4500, 1000, 1000
    pointsInTsClass=800
    pointsInTeClass= ((1000, 500, 250, 187, 125, 87, 50, 25, 15, 10))
    IMG_SIZE = 32
    h, w, c = 32, 32, 3
    numClass = 10
elif ds == 'celeba':
    home_dir = Path.home()
    data_dir = home_dir / 'datasets/CelebA'
    attr_of_interest = ('Blond_Hair', 'Black_Hair', 'Bald', 'Brown_Hair', 'Gray_Hair')
    SetOri = np.load(str(data_dir / 'celeba_imb_haircolor_data.npz'))['arr_0']
    LabOri = np.load(str(data_dir / 'celeba_imb_haircolor_data.npz'))['arr_1']
    rnd_idx = np.random.default_rng().choice(len(SetOri), size=len(SetOri), replace=False)
    train_idx, test_idx = rnd_idx[:80000], rnd_idx[80000:]
    trainSetOri, trainLabOri = SetOri[train_idx], LabOri[train_idx]
    testSetOri, testLabOri = SetOri[test_idx], LabOri[test_idx]
    pointsInTrClass=((15000, 1500, 750, 300, 150))
    maxPTrClass, maxPTsClass, maxPTeClass = 15000, 1000, 1000
    pointsInTsClass=800
    pointsInTeClass= ((1000, 500, 111, 55, 17))
    IMG_SIZE = 64
    h, w, c = 64, 64, 3
    numClass = 5

classLocTr=np.insert(np.cumsum(pointsInTrClass), 0, 0)
classMapTr, classMapTs, trainPoints, testPoints=list(), list(), list(), list()
for i in range(numClass):
    classMapTr.append(np.where(trainLabOri==i)[0])
    classMapTs.append(np.where(testLabOri==i)[0])
trainS=np.zeros((np.sum(pointsInTrClass), trainSetOri.shape[1], trainSetOri.shape[2], trainSetOri.shape[3]))
trainL=np.zeros((np.sum(pointsInTrClass),1))

for i in range(numClass):
    # randIdxTr=np.random.randint(0, maxPTrClass, pointsInTrClass[i])
    randIdxTr=np.random.default_rng().choice(min(maxPTrClass, len(classMapTr[i])), size=pointsInTrClass[i], replace=False)
    trainPoints.append(classMapTr[i][randIdxTr])
    trainS[classLocTr[i]:classLocTr[i+1]]=trainSetOri[trainPoints[i]]
    trainL[classLocTr[i]:classLocTr[i+1], 0]=trainLabOri[trainPoints[i]]

testS=np.zeros((int(numClass*pointsInTsClass), trainSetOri.shape[1], trainSetOri.shape[2], trainSetOri.shape[3]))
testL=np.zeros((int(numClass*pointsInTsClass),1))
classLocTs=np.arange(0, (numClass+1)*pointsInTsClass, pointsInTsClass)
for i in range(numClass):
    # randIdxTs=np.random.randint(0, maxPTsClass, pointsInTsClass)
    randIdxTs=np.random.default_rng().choice(maxPTsClass, size=pointsInTsClass, replace=False)
    testPoints.append(classMapTs[i][randIdxTs])
    testS[classLocTs[i]:classLocTs[i+1]]=testSetOri[testPoints[i]]
    testL[classLocTs[i]:classLocTs[i+1], 0]=testLabOri[testPoints[i]]


classLocTe=np.insert(np.cumsum(pointsInTeClass), 0, 0)
classMapTe, tePoints=list(), list()
for i in range(numClass):
    classMapTe.append(np.where(testLabOri==i)[0])
imbTestS=np.zeros((np.sum(pointsInTeClass), trainSetOri.shape[1], trainSetOri.shape[2], trainSetOri.shape[3]))
imbTestL=np.zeros((np.sum(pointsInTeClass),1))

for i in range(numClass):
    randIdxTe=np.random.default_rng().choice(min(maxPTeClass, len(classMapTe[i])), size=pointsInTeClass[i], replace=False)
    tePoints.append(classMapTe[i][randIdxTe])
    imbTestS[classLocTe[i]:classLocTe[i+1]]=testSetOri[tePoints[i]]
    imbTestL[classLocTe[i]:classLocTe[i+1], 0]=testLabOri[tePoints[i]]


np.savez(f'{data_dir}/{ds}.npz', 
         trainS=trainS, trainL=trainL.flatten(), testS=testS, testL=testL.flatten(), imbTestS=imbTestS, imbTestL=imbTestL.flatten())

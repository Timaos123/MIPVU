#coding:utf8
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Bidirectional, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from keras.utils.generic_utils import CustomObjectScope
import os
import numpy as np
import pandas as pd
import pickle as pkl
import keras.backend as K
from gensim.models import Word2Vec
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def getRecall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def getPrecision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


class ACGAN():
    def __init__(self, sequenceList, vocabSize, num_classes, labelEmlabelDict, myTokenizer, w2vModel, vecSize=100, windowSize=3):
        # Input shape
        seqLen = max([len(seqItem) for seqItem in sequenceList])
        self.vocabSize = vocabSize
        self.vecSize = vecSize
        self.seqLen = seqLen
        self.channels = 1
        self.windowSize = windowSize
        self.seqShape = (self.vecSize, self.seqLen, self.channels)
        self.CNNKernelShape = (self.windowSize, self.vecSize)
        self.num_classes = num_classes+1
        self.latent_dim = 100
        self.tokenizer = myTokenizer
        self.w2vModel = w2vModel
        self.labelEmlabelDict = labelEmlabelDict
        self.vocabArr = np.array([w2vModel.wv[word]
                                  for word in w2vModel.wv.vocab])
        self.indexWordDict = dict(list(
            zip(list(self.tokenizer.word_index.values()), list(self.tokenizer.word_index.keys()))))

        self.discriminator = self.build_discriminator()

        wordInput = Input(shape=(1, self.vecSize,), name="wordInput")
        meanInput = Input(shape=(self.seqLen, self.vecSize), name="meanInput")
        self.generator = self.build_generator()

        fakeExam = self.generator([wordInput, meanInput])
        valid, word = self.discriminator(fakeExam)
        self.combined = Model([wordInput, meanInput], [valid, word])
        self.combined.summary()

        self.combined.compile(loss=["binary_crossentropy", "categorical_crossentropy"], optimizer="Adam", metrics=[
                              "acc", getRecall, getPrecision])

    def build_generator(self):
        wordInput = Input(shape=(1, self.vecSize), name="wordInput")

        meanInput = Input(shape=(self.seqLen, self.vecSize,), name="meanInput")
        biLSTMLayer = LSTM(units=self.vecSize, return_sequences=False, name="encoder")(
            meanInput)

        # doubleWordEmbeddingLayer=concatenate([wordEmbeddingFlattenLayer, wordEmbeddingFlattenLayer],axis=1)
        concateLayer = multiply([biLSTMLayer, wordInput])
        # concateLayer=Reshape((1,int(concateLayer.shape[1])))(concateLayer)

        biLSTMLayerList = []
        biLSTMLayer = Bidirectional(
            LSTM(units=self.vecSize, return_sequences=True, name="decoder"))(concateLayer)
        biLSTMLayerList.append(biLSTMLayer)
        for i in range(self.seqLen-1):
            biLSTMLayer = Bidirectional(
                LSTM(units=self.vecSize, return_sequences=True, name="decoder"))(biLSTMLayer)
            biLSTMLayerList.append(biLSTMLayer)
        biLSTMFinalLayer = concatenate(biLSTMLayerList, axis=1)
        timeDenseLayer = TimeDistributed(
            Dense(units=self.vecSize))(biLSTMFinalLayer)

        myModel = Model([wordInput, meanInput],
                        timeDenseLayer, name="generator")
        myModel.summary()
        return myModel

    def build_discriminator(self):
        inputLayer = Input(shape=(self.seqLen, self.vecSize))
        setenceLantLayer = Dense(units=1)(inputLayer)
        flattenLayer = Flatten()(setenceLantLayer)
        validOutDense = Dense(units=1, name="validOutDense")(flattenLayer)
        wordOutDense = Dense(units=self.vocabSize,
                             name="wordOutDense")(flattenLayer)

        disModel = Model(
            inputLayer, [validOutDense, wordOutDense], name="discriminator")
        disModel.summary()
        disModel.compile("Adam", loss="mse", metrics=[
                         "acc", getRecall, getPrecision])

        return disModel

    def train(self, X_train, y_trainArr, y_train, means, epochs, batch_size=128, sample_interval=50, rebuildData=False):
        #check if batch_size is too large
        if batch_size > X_train.shape[0]:
            batch_size = X_train.shape[0]
        indexArr = np.array([i for i in range(X_train.shape[0])])
        labelSetList = list(set(y_train.T.tolist()[0]))
        for epoch in range(epochs):

            #adjust the discriminator's trainable
            self.discriminator.trainable = False

            #sampling
            np.random.shuffle(indexArr)
            exams = X_train[indexArr]
            wordLabels = y_train[indexArr]
            meanSample = means[indexArr]
            #embedded words, one-hot embedded words, word labels,meansample,exams
            if rebuildData == True:
                embeddedOneHotWordLabels = np.array([[self.w2vModel.wv[str(
                    self.tokenizer.word_index[y_train[indexItem]])], self.tokenizer.texts_to_matrix(y_train[indexItem])[0], wordLabels[indexItem], meanSample[indexItem], exams[indexItem]] for indexItem in indexArr[:batch_size] if y_train[indexItem] in self.tokenizer.word_index.keys()])
                with open("data/embeddedOneHotWordLabels.pkl", "wb+") as embeddedOneHotWordLabelsFile:
                    pkl.dump(embeddedOneHotWordLabels,
                             embeddedOneHotWordLabelsFile)
            else:
                with open("data/embeddedOneHotWordLabels.pkl", "rb") as embeddedOneHotWordLabelsFile:
                    embeddedOneHotWordLabels = pkl.load(
                        embeddedOneHotWordLabelsFile)
            embededWordLabels = embeddedOneHotWordLabels[:, 0].reshape(
                (embeddedOneHotWordLabels.shape[0], 1, -1))
            embededWordLabels = np.array([row[0][0] for row in embededWordLabels]).reshape(
                (embeddedOneHotWordLabels.shape[0], 1, -1))

            oneHotWordLabel = embeddedOneHotWordLabels[:, 1]
            oneHotWordLabel=np.array([row for row in oneHotWordLabel])

            wordLabels = embeddedOneHotWordLabels[:, 2]

            meanSample = embeddedOneHotWordLabels[:, 3]
            meanSample = np.array([row for row in meanSample])

            exams = embeddedOneHotWordLabels[:, 4]
            exams = np.array([row for row in exams])

            realBatchSize=exams.shape[0]
            valid = np.ones((realBatchSize, 1))
            fake = np.zeros((realBatchSize, 1))

            #generalizing fake samples
            fakeExams = self.generator.predict(
                [embededWordLabels, meanSample])

            #training the discriminator
            metricsList = self.discriminator.metrics_names
            d_loss_real = self.discriminator.train_on_batch(
                exams, [valid, oneHotWordLabel])
            d_loss_fake = self.discriminator.train_on_batch(
                fakeExams, [fake, oneHotWordLabel])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            metricsDict = dict([(metricsList[i], d_loss[i])
                                for i in range(len(metricsList))])
            #adjust the discriminator's trainable
            self.discriminator.trainable = False

            #training generator's loss
            g_loss = self.combined.train_on_batch(
                [embededWordLabels, meanSample], [fake, oneHotWordLabel])
            print(epoch, ":", "d:", metricsDict, "g:loss:", g_loss[0])

    def genPredictItem(self, inputList):
        #predict with single word and its mean
        sourceWord = inputList[0]
        sourceMean = inputList[1]
        sourceWordIndexArr = np.array(
            [self.w2vModel.wv[str(self.tokenizer.word_index[sourceWord])]]).reshape(-1, 1, 100)
        sourceMeanIndexList = [[self.tokenizer.word_index[wordItem] for wordItem in sourceMean.split(
            " ") if wordItem in list(self.tokenizer.word_index.keys())]]
        sourceMeanIndexList = pad_sequences(
            sourceMeanIndexList, self.seqLen, padding="post")
        sourceMeanIndexList = [[self.w2vModel.wv[str(
            wordItem)] for wordItem in row] for row in sourceMeanIndexList]
        sourceMeanIndexArr = np.array(
            sourceMeanIndexList).reshape(-1, self.seqLen, self.vecSize)
        sentenceVecArr = self.generator.predict(
            [sourceWordIndexArr, sourceMeanIndexArr])
        sentence = [" ".join([self.indexWordDict[self.sentVec2Word(
            wordItem)] for wordItem in row if self.sentVec2Word(
            wordItem) != 0]) for row in sentenceVecArr][0]
        return sentence

    def sortByPos(self, sentence):
        '''
        sort a sentence according to the distribution of sentences
        '''
        pass

    def sentVec2Word(self, wordArr):
        '''
        wordArr:3-D vector
        '''
        vocabSqArr = np.sum(self.vocabArr*self.vocabArr, axis=1).T
        wordSq = np.dot(wordArr, wordArr.T)
        vocabWordDisArr = np.sqrt(
            (self.vocabArr-wordArr)*(self.vocabArr-wordArr))
        vocabWordDisArr = np.sum(vocabWordDisArr*vocabWordDisArr, axis=1).T
        vocabWordArr = np.dot(self.vocabArr, wordArr.T).T
        disList = ((vocabSqArr+wordSq-vocabWordDisArr)/2*vocabWordArr).tolist()
        chosenIndex = disList.index(max(disList))
        return chosenIndex


if __name__ == '__main__':

    vecSize = 100
    topN = -1
    rebuildData = True
    loadModel = False
    epochs=1500

    trainDf = pd.read_csv("data/GANData.csv")
    trainDf["xTrain"] = trainDf["exam"]
    trainDf["yTrain"] = trainDf["keyWord"]

    print("Load the dataset ...")
    X_train = np.array(trainDf["xTrain"])[:topN].tolist()
    y_train = np.array(trainDf["yTrain"])[:topN]
    means = np.array(trainDf["mean"])[:topN].tolist()

    print("Configure inputs ...")
    myTokenizer = Tokenizer(lower=True, split=" ")
    myTokenizer.fit_on_texts(X_train)
    X_train = myTokenizer.texts_to_sequences(X_train)
    seqLen = max([len(XRow) for XRow in X_train])
    vocabSize = len(myTokenizer.word_index.keys())
    X_train = pad_sequences(X_train, maxlen=seqLen, padding="post")
    X_train = [[str(wordItem) for wordItem in row]
               for row in X_train]
    myW2VModel = Word2Vec(X_train, size=vecSize, min_count=0)
    myW2VModel.train(
        X_train, total_examples=myW2VModel.corpus_count, epochs=myW2VModel.epochs)
    X_train = np.array([[myW2VModel.wv[wordItem]
                         for wordItem in row] for row in X_train])

    means = myTokenizer.texts_to_sequences(means)
    means = pad_sequences(means, maxlen=seqLen, padding="post")
    means = [" ".join([str(indexItem) for indexItem in row]) for row in means]
    meanVecList = [[myW2VModel.wv[wordItem] for wordItem in row if wordItem in list(
        myW2VModel.wv.vocab)][0:seqLen] for row in means]

    print("padding meanVecList")
    for rowI in range(len(meanVecList)):
        while len(meanVecList[rowI]) < seqLen:
            meanVecList[rowI].append(myW2VModel.wv["0"])
    meanVecArr = np.array(meanVecList)

    print("configuring classes")
    y_trainArr = np.array(y_train).T
    myClassTokenizer = Tokenizer(lower=True, split=" ")
    myClassTokenizer.fit_on_texts(y_trainArr)
    y_trainArr = myClassTokenizer.texts_to_matrix(y_trainArr)
    labelEmlabelDict = dict(list(zip(y_train, y_trainArr)))
    num_classes = len(set(y_train.tolist()))

    print("training the generator")
    if loadModel==True:
        #if you have already saved the model
        with CustomObjectScope({"getRecall": getRecall, "getPrecision": getPrecision}):
            with open("model/acganModel.model", "rb") as acganModelFile:
                acgan = pkl.load(acganModelFile)
    else:
        #else
        acgan = ACGAN(X_train, vocabSize+1, num_classes,
                      labelEmlabelDict, myTokenizer, myW2VModel, vecSize=vecSize)
        acgan.train(X_train, y_trainArr, y_train,  meanVecArr, epochs=epochs,
                    batch_size=32, sample_interval=200, rebuildData=rebuildData)
        with open("model/acganModel.model", "wb+") as acganModelFile:
            pkl.dump(acgan, acganModelFile)

    print("testing the generator")
    testSample = [
        "abandoned", "left alone by someone who should stay with you and look after you"]
    print(acgan.genPredictItem(testSample))
    input("press any key to escape ...")

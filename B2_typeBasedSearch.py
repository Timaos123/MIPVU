#coding:utf8
'''
Created on 2018年12月16日

@author: Administrator
'''
import tryCrawl as tc
import tryGetDistanceBetweenSentences as tgbs
import re
import numpy as np
import string
from threading import Thread
import tqdm
import tryLoadWordVec as tlwv
import nltk.stem
import pickle as pkl
from keras.utils.generic_utils import CustomObjectScope
import keras.backend as K
import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

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

def getWordMeanPos(wordItem,mySentence,WVModel):
    print("loading gan model")
    #if you have already saved the model
    with CustomObjectScope({"getRecall": getRecall, "getPrecision": getPrecision}):
        with open("model/acganModel.model", "rb") as acganModelFile:
            acgan = pkl.load(acganModelFile)
    wordMeanExamList=tc.main(wordItem,acgan)
    wordMeanExamDistanceList=[]
    if len(wordMeanExamList)==0:
        return -1
    for wordMeanExamItem in wordMeanExamList:
        examItemDistanceList=[]
        for examItem in wordMeanExamItem[1]:
            examItemDistanceItem=tgbs.getDistanceBetweenSentences(mySentence,examItem,WVModel)
            examItemDistanceList.append(examItemDistanceItem)
        examItemDistance=np.mean(examItemDistanceList)
        wordMeanExamDistanceList.append(examItemDistance)
    meanPos=wordMeanExamDistanceList.index(min(wordMeanExamDistanceList))
    return meanPos

def main(nodeWord,mySentence):
    s = nltk.stem.SnowballStemmer('english')
    if s.stem(nodeWord) not in [s.stem(wordItem) for wordItem in mySentence.split(" ")]:
        return "err:no that word"
    myWVModel=tlwv.loadWordVec()
    mySentence=mySentence.lower()
    mySentence=re.sub('['+string.punctuation+']','',mySentence)
    meanPos=getWordMeanPos(nodeWord,mySentence,myWVModel)
    if meanPos==0:return False
    else: return True
    
if __name__ == '__main__':
    mySentence="the car is like his son"
    nodeWord="cat"
    print(main(nodeWord,mySentence))

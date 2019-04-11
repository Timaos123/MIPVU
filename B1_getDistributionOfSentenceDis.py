#coding:utf8

import pandas as pd
import tryCrawl as tc
import tryGetDistanceBetweenSentences as tgbs
import re
import numpy as np
import string
from threading import Thread
import tqdm
import tryLoadWordVec as tlwv
import nltk.stem
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
import pickle as pkl

def getSentenceBasicMeanDis(keyWord,sentence,w2vModel):
    '''
    input:
    keyWord(str):input word
    sentence(str):keyWord's context
    w2vModel(gensim.models.Word2vec.model):presaved w2v model
    -------------------------------
    return:
    distance(float): mean float of the sentence and the basic mean
    '''
    meanExamList=tc.main(keyWord)
    if len(meanExamList)>0:
        basicMeanExamList=meanExamList[0][1]
        basicMeanExamList=[[word for word in basicMeanExamItem.split(" ") if word in w2vModel.wv.vocab] for basicMeanExamItem in basicMeanExamList ]
        sentence=[word for word in sentence.split(" ") if word in w2vModel.wv.vocab]
        distanceList=[w2vModel.n_similarity(sentence,exam) for exam in basicMeanExamList]
        return np.mean(distanceList)
    return 0

if __name__=="__main__":
    print("loading data ...")
    corpusDf=pd.read_csv("data/structuredData.csv")
    TcorpusDf=corpusDf[corpusDf["type"]==True]
    FcorpusDf=corpusDf[corpusDf["type"]==False]
    print("T samples:",TcorpusDf.shape[0])
    print("F samples:",FcorpusDf.shape[0])
    
    print("loading model ...")
    myW2VModel=Word2Vec.load("model/w2vModel.model")
    # print(getSentenceBasicMeanDis("fund","we need to sell the fund",myW2VModel))

    print("transforming data ...")
    TcorpusArr=np.array(TcorpusDf)
    FcorpusArr=np.array(FcorpusDf)
    TcorpusBasDisList=[getSentenceBasicMeanDis(row[1],row[3],myW2VModel) for row in tqdm.tqdm(TcorpusArr)]
    with open("data/TcorpusBasDisList.pkl","wb+") as TcorpusBasDisListFile:
        pkl.dump(TcorpusBasDisList,TcorpusBasDisListFile)
    FcorpusBasDisList=[getSentenceBasicMeanDis(row[1],row[3],myW2VModel) for row in tqdm.tqdm(FcorpusArr)]
    with open("data/FcorpusBasDisList.pkl","wb+") as FcorpusBasDisListFile:
        pkl.dump(FcorpusBasDisList,FcorpusBasDisListFile)

    print("getting distribution ...")
    TMean=np.mean(TcorpusBasDisList)
    FMean=np.mean(FcorpusBasDisList)
    TStd=np.std(TcorpusBasDisList)
    FStd=np.std(FcorpusBasDisList)
    print("distribution of T:",TMean,TStd)
    print("distribution of F:",FMean,FStd)
    TFDistributionDict={"TMean":TMean,"TStd":TStd,"FMean":FMean,"FStd":FStd}
    with open("model/TFDistributionDict.pkl","wb+") as TFDistributionDictFile:
        pkl.dump(TFDistributionDict,TFDistributionDictFile)
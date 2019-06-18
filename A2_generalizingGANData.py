#coding:utf8
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tryCrawl as TC
import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import nltk.stem.snowball as sb
import re

if __name__=="__main__":
    '''
    developing GAN data: {word1:[(mean1,[exam1,...]),
                                 (mean2,[exam1,...])],
                          word2:[(mean1:[exam1,...]),
                                 (mean2:[exam1,...])}
    '''
    print("loading data ...")
    sourceDf=pd.read_csv("data/structuredData.csv")
    
    print("abstract corpus ...")
    corpusList=np.array(sourceDf["message"]).tolist()
    print("getting vocab list ...")
    vectorizer=CountVectorizer(min_df=0)
    vectorizer.fit(corpusList)
    vocabList=vectorizer.get_feature_names()
    
#     wordFqArr=np.array([word[1] for word in vectorizer.vocabulary_.items()])
#     wordFqArr=(wordFqArr-np.mean(wordFqArr)*np.ones(wordFqArr.shape))/np.std(wordFqArr)
#     print(list(vectorizer.vocabulary_.items()))
#     print("mean of word frequence:",np.mean(wordFqArr))
#     print("standard variance of word frequence:",np.std(wordFqArr))
#     plt.hist(wordFqArr)
#     plt.show()

    print("getting mean-example list ...")
    meanExamList=[]
    wordMeanExamDict={}
    sb_stemmer=sb.SnowballStemmer("english")
    for word in tqdm.tqdm(vocabList):
        if len(re.findall("[0-9]+",word))>0:
            continue
        if word not in wordMeanExamDict.keys():
            wordMeanExamDict[word]=TC.main(word.strip())
        stemedWord=sb_stemmer.stem(word)
        if stemedWord not in wordMeanExamDict.keys():
              wordMeanExamDict[stemedWord]=TC.main(stemedWord.strip())

    print("saving data ...")
    with open("data/GANDict.pkl","wb+") as GANDictFile:
        pkl.dump(wordMeanExamDict,GANDictFile)
    
    print("loading data ...")
    with open("data/GANDict.pkl","rb") as GANDictFile:
        GANDict=pkl.load(GANDictFile)
    
    print("configuration ...")
    GANDataList=[]
    for keyItem in GANDict.keys():
        for meanExamItem in GANDict[keyItem]:
            for examItem in meanExamItem[1]:
                GANDataList.append([keyItem,meanExamItem[0],examItem])
    GANDataDf=pd.DataFrame(np.array(GANDataList),columns=["keyWord","mean","exam"])

    print("saving configured data ...")
    GANDataDf.to_csv("data/GANData.csv")

    print("finished !")
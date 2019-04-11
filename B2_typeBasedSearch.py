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

def getWordMeanPos(wordItem,mySentence,WVModel):
    s = nltk.stem.SnowballStemmer('english')
    wordItem=s.stem(wordItem)
    wordMeanExamList=tc.main(wordItem)
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
    if nodeWord not in mySentence.split(" "):
        return "err:no that word"
    myWVModel=tlwv.loadWordVec()
    mySentence=mySentence.lower()
    mySentence=re.sub('['+string.punctuation+']','',mySentence)
    myWordList=mySentence.split(" ")
    meanPos=getWordMeanPos(nodeWord,mySentence,myWVModel)
    if meanPos==0:return False
    else: return True
    
if __name__ == '__main__':
    mySentence="the car is like his son"
    nodeWord="cat"
    print(main(nodeWord,mySentence))

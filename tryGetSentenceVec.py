#coding:utf8
'''
Created on 2018年12月12日

@author: Administrator
'''
import string
import numpy as np
import tryGetWordVec as TGWV
import re
from gensim.models import Word2Vec

def getSentenceVec(mySentence,myW2VModel):
    mySentence=mySentence.lower()
    mySentence=re.sub('['+string.punctuation+']','',mySentence)
    myWordList=mySentence.split(" ")
    for myWordItem in myWordList:
        try:
            myWordList.remove("")
        except ValueError:
            break
    myWordVecArr=np.array([myW2VModel.wordIndexDict[wordItem] for wordItem in myWordList if wordItem in myW2VModel.wordIndexDict])
    return myWordVecArr

if __name__ == '__main__':
    mySentence="We might need to sell this fund."
    myW2VModel=Word2Vec.load("model/w2vModel.model")
    print(getSentenceVec(mySentence,myW2VModel))
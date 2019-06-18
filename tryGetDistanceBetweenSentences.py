#coding:utf8
'''
Created on 2018年12月12日
@author: Administrator
'''

import tryGetSentenceVec as TGSV
import numpy as np
import tqdm
from nltk.corpus import wordnet as wn

def getDistanceBetweenSentences(sent1,sent2,WVModel=None,vectorType="VDB"):
    disList=[]      
    if vectorType=="VDB":
        sent1V=TGSV.getSentenceVec(sent1,WVModel)
        sent2V=TGSV.getSentenceVec(sent2,WVModel)
        for row1 in sent1V:
            for row2 in sent2V:
                disList.append(np.linalg.norm(row1-row2))
        ave=0
        disI=0
        for disItem in disList:
                ave+=disItem
                disI+=1
        if disI==0:
                disI=1
        ave=ave/disI
    elif vectorType=="LDB":
        synList1=[wn.synsets(wordItem)[0] for wordItem in sent1.split(" ") if len(wn.synsets(wordItem))>0]
        synList2=[wn.synsets(wordItem)[0] for wordItem in sent2.split(" ") if len(wn.synsets(wordItem))>0]
        distanceArr=np.array([[synItem1.path_similarity(synItem2) for synItem2 in synList2] for synItem1 in synList1])
        distanceArr[distanceArr==None]=np.inf
        distanceArr[distanceArr<0]=np.inf
        
        ave=np.mean(distanceArr[distanceArr!=np.inf])
    return ave

if __name__ == '__main__':
    print(getDistanceBetweenSentences("you are like a dog~","a dog is better than you!",vectorType="LDB"))
    print(getDistanceBetweenSentences("I love you","a beautiful beach",vectorType="LDB"))
    print(getDistanceBetweenSentences("you are like a dog~","you are like a dog~",vectorType="LDB"))
    print(getDistanceBetweenSentences("you are like a dog~","you are like a dog~ you are like a dog~",vectorType="LDB"))
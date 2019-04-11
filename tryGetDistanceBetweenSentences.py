#coding:utf8
'''
Created on 2018年12月12日
@author: Administrator
'''

import tryGetSentenceVec as TGSV
import numpy as np
import tqdm

def getDistanceBetweenSentences(sent1,sent2,WVModel):
    disList=[]
    sent1V=TGSV.getSentenceVec(sent1,WVModel)
    sent2V=TGSV.getSentenceVec(sent2,WVModel)
    for row1 in sent1V:
        for row2 in sent2V:
#             num=row1.T*row2
#             denum=np.linalg.norm(row1) * np.linalg.norm(row2) 
#             cos=num/denum
#             disList.append(cos)
            disList.append(np.linalg.norm(row1-row2))
    ave=0
    disI=0
    for disItem in disList:
        ave+=disItem
        disI+=1
    if disI==0:
        disI=1
    ave=ave/disI
    return ave
    
if __name__ == '__main__':
    print(getDistanceBetweenSentences("you are like a dog~","a dog is better than you!"))
    print(getDistanceBetweenSentences("I love you","a beautiful beach"))
    print(getDistanceBetweenSentences("you are like a dog~","you are like a dog~"))
    print(getDistanceBetweenSentences("you are like a dog~","you are like a dog~ you are like a dog~"))
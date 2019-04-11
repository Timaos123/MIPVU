#coding:utf8
'''
Created on 2018年12月12日

@author: Administrator
'''
import os
import time
import pickle as pkl
def getWordVec(myWord,dim=100):
    for fileNameItem in os.listdir("glove.27B"):
        if fileNameItem=="glove.twitter.27B."+str(dim)+"d.txt":
            with open(os.path.join("glove.27B",fileNameItem),"r",encoding="utf8") as preTrainedVectorFile:
                wordList=preTrainedVectorFile.readlines()
                for wordRow in wordList:
                    word=wordRow.split(" ")[0]
                    if word==myWord:
                        return [float(dim) for dim in wordRow.split(" ")[1:]]
    return [0 for i in range(100)]
if __name__ == '__main__':
    myWord="dog"
    startTime=time.time()
    getWordVec(myWord)
    endTime=time.time()
    print("read time:",endTime-startTime)
    
            
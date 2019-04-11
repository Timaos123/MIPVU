#coding:utf8

import pickle as pkl
import os
import time
import tqdm
import re

class wordVecModel:
    def __init__(self,wordIndexDict,wordVecList):
        self.wordIndexDict=wordIndexDict
        self.wordVecList=wordVecList
        
    def wordVec(self,word):
        try:
            return self.wordVecList[self.wordIndexDict[word]]
        except KeyError:
            return [0 for i in range(100)]

def loadWordVec():
    try:
        with open("model/vectorList.pkl","rb") as vecFile:
            wordVecList=pkl.load(vecFile)
        with open("model/wordDict.pkl","rb") as wordDictFile:
            wordIndexDict=pkl.load(wordDictFile)
    except:
        i=0
        wordDictList=[]
        wordVecList=[]
        for fileNameItem in os.listdir("glove.27B"):
            if fileNameItem=="glove.twitter.27B."+str(100)+"d.txt":
                with open(os.path.join("glove.27B",fileNameItem),"r",encoding="utf8") as preTrainedVectorFile:
                    wordList=preTrainedVectorFile.readlines()
                    for wordRow in tqdm.tqdm(wordList):
                        try:
                            word=wordRow.split(" ")[0]
                            wordRow.replace("- ","-")
                            wordRow=re.sub("\-\s+","\-",wordRow)
                            wordVec=[float(dim) for dim in wordRow.split(" ")[1:]]
                            i+=1
                            wordDictObj=[word,i]
                            wordDictList.append(wordDictObj)
                            wordVecList.append(wordVec)
                        except ValueError:
                            print("err:",wordRow.split(" ")[0])
        wordIndexDict=dict(wordDictList)
        if "model" not in os.listdir("."):
            os.mkdir("model")
        with open("model/vectorList.pkl","wb+") as vecFile:
            pkl.dump(wordVecList,vecFile)
        with open("model/wordDict.pkl","wb+") as wordDictFile:
            pkl.dump(wordIndexDict,wordDictFile)
    return wordVecModel(wordIndexDict,wordVecList)

if __name__=="__main__":
    start=time.time()
    word="dog"
    myWVModel=loadWordVec()
    mid=time.time()
    print("loading time:",mid-start)
    myWVModel.wordVec(word)
    end=time.time()
    print("end time:",end-mid)
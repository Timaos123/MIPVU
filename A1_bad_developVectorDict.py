#coding:utf8
'''
Created on 2018年12月13日

@author: Administrator
'''

import os
import pickle as pkl
import tqdm

if __name__ == '__main__':
    rootPath="glove.6B"
    
    wordList=[]
    vectorList=[]
    print("reading vectors ...")
    for fileNameItem in tqdm.tqdm(os.listdir(rootPath)):
        print("dealing with",fileNameItem)
        with open(os.path.join(rootPath,fileNameItem),"r",encoding="utf8") as vectorFile:
            rowList=vectorFile.readlines()
            wordList=wordList+[row.split(" ")[0] for row in rowList]
            vectorList=vectorList+[[float(dimItem) for dimItem in row.split(" ")[1:]] for row in rowList]
    wvDict=dict(list(zip(wordList,vectorList)))
    
    if "vectorFile" not in os.listdir("."):
        os.mkdir("vectorFile")
        
    print("saving data ...")
    with open("vectorFile/vectorFile.pkl","wb+") as vectorFile:
        pkl.dump(vectorList,vectorFile)
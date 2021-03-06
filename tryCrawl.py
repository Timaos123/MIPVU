#coding:utf8
'''
Created on 2018年12月6日

@author: Administrator
'''
import urllib.request as request
from bs4 import BeautifulSoup as bs
import json
import tqdm
import re
import nltk.stem.snowball as sb
def main(word,keepNone=False):
    '''
    input:
    word(str):input word
    ---------------------
    return:
    meanExamList(list): [(mean1:[mean1exm1,mean1exm2,..]),(mean2,[mean2exam1,...])]
    '''
    print("connecting to the website ...")
    myUrl="https://www.macmillandictionary.com/dictionary/british/"+word
    res=request.urlopen(myUrl)
    res.encoding = 'utf-8'
    
    print("finding needed information ...")
    soupStr=bs(res.read(),features="lxml")
    olBsL=soupStr.find_all("ol",class_="senses")
    if len(olBsL)>0:
        olBs=olBsL[0]
    else:
        print("problems in '",word,"':no enough meanings")
        tempWord=word
        sb_stemmer=sb.SnowballStemmer("english")
        word=sb_stemmer.stem(word)
        if tempWord!=word:
            return main(word)
        else:
            return []
    liBsList=olBs.find_all("li")
    
    print("finding meanings and examples ...")
    meaningList=[]
    exampleList=[]
    for liItem in liBsList:
        if len(liItem.find_all("div",class_="SENSE"))>0:
            if len(liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("span",class_="DEFINITION"))>0:
                liDivItem=liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("span",class_="DEFINITION")[0].\
                    text
            elif len(liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("span",class_="GREF-ENTRY"))>0:
                liDivItem=liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("span",class_="GREF-ENTRY")[0].\
                    find_all("a")[0].text
                if liDivItem==None:
                    print("problems in",word,": None")
                    return [("no means","no examples")]
                else:
                    return main(liDivItem,keepNone=False)
            elif len(liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("div",class_="sideboxbody"))>0:
                liDivItem=liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("div",class_="sideboxbody")[0].\
                    find_all("a")[0].text
                if liDivItem==None:
                    print("problems in",word,": None")
                    return [("no means","no examples")]
                else:
                    return main(liDivItem,keepNone=False)
            else:
                return [("no means","no examples")]
            meaningList.append(liDivItem)
        if len(liItem.find_all("div",class_="SENSE"))>0:
            try:
                liDivItem=liItem.\
                    find_all("div",class_="SENSE")[0].\
                    find_all("p",class_="EXAMPLE")
                exampleList.append([liDivItemItem.text for liDivItemItem in liDivItem])
            except IndexError:
                pass
    meanExamList=list(zip(meaningList,exampleList))
    if len(meanExamList[0][1])==0:
        print("problems in '",word,"':no enough examples. replacing examples with meanings")
        meanExamList=[(meanExamItem[0],[meanExamItem[0]]) for meanExamItem in meanExamList]
    tempMeanExamList=[]
    if keepNone==False:
        for row in meanExamList:
            if len(row[1])!=0:
                tempMeanExamList.append(row)
        meanExamList=tempMeanExamList
    return meanExamList
    
if __name__ == '__main__':
    print(main("kept",keepNone=False))
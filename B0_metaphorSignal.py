#coding:utf8
'''
Created on 2018年12月13日

@author: Administrator
'''

from nltk.corpus import wordnet as wn

def checkWhetherMetaphor(mySentence):
    similarWordList=[]
    for synset0 in wn.synsets('similar'):
        similarWordList+=synset0.lemma_names()
        for word in synset0.lemma_names():
            for synset1 in wn.synsets(word):
                similarWordList+=synset1.lemma_names()
    similarWordList=list(set(similarWordList))
    for sentenceWord in mySentence.split():
        if sentenceWord in similarWordList:
            return True
    return False
    
if __name__ == '__main__':
    mySentence="you are similar to a dog."
    print(checkWhetherMetaphor(mySentence))
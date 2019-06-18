#codint:utf8
import tryVisNorm as tvm
import pickle as pkl
import numpy as np
if __name__=="__main__":
    with open("data/FcorpusBasDisList.pkl","rb") as corpusBasDisListFile:
        corpusBasDisList=pkl.load(corpusBasDisListFile)
    myPlt=tvm.visNormal(np.array(corpusBasDisList),deleteOutlier=False,partNum=10)
    myPlt.show()
#coding:utf8

import B0_metaphorSignal as B0
import B2_typeBasedSearch as B2
from sklearn.metrics import accuracy_score
import tqdm
import pandas as pd
import numpy as np
from A4_trainGAN import *

if __name__=="__main__":
    sampleSize=10
    myDataDf=pd.read_csv("data/structuredData.csv")
    myDataList=np.array(myDataDf.loc[:,["message","type","nodeWord"]].sample(sampleSize)).tolist()
    preY=[B0.checkWhetherMetaphor(preItem[0]) or B2.main(preItem[2],preItem[0]) for preItem in myDataList]
    trueY=[preItem[1] for preItem in myDataList]
    acc=accuracy_score(trueY,preY)
    print(trueY)
    print(preY)
    print("acc:",acc)
    

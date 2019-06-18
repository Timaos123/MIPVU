#coding:utf8

import B0_metaphorSignal as B0
import B1_typeBasedSearch as B1
from sklearn.metrics import accuracy_score
import tqdm
import pandas as pd
import numpy as np

if __name__=="__main__":
    myDataDf=pd.read_csv("data/structuredData.csv")
    myDataList=np.array(myDataDf.loc[:,["message","type"]]).tolist()[:10]
    preY=[B0.checkWhetherMetaphor(preItem[0]) or B1.main(preItem[0]) for preItem in myDataList]
    trueY=[preItem[1] for preItem in myDataList]
    acc=accuracy_score(trueY,preY)
    print("acc:",acc)
    

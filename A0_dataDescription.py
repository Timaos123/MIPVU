#coding:utf8
import pandas as pd
if __name__=="__main__":
    structuredDf=pd.read_csv("data/structuredData.csv")
    
    distributionDf=structuredDf.groupby("type").size()
    print(distributionDf)
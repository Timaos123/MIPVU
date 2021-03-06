#coding:utf8
'''
Created on 2018年12月15日

@author: Administrator
'''

import pandas as pd
import tqdm
import numpy as np
import re


def transTF(YN):
    '''
    input:
    transform Y/N into True/False
    YN: the series filled with Y/N
    return:
    True/False
    '''
    if YN == "Y":
        return True
    else:
        return False


if __name__ == '__main__':

    print("loading data ...")
    excelFile = pd.ExcelFile("data/metaphor_detection.xlsx")
    corpusDf = pd.read_excel("data/metaphor_detection.xlsx")

    keyList = ["nodeWord"]+list(corpusDf.keys())
    corpusDf = pd.DataFrame(columns=keyList)
    for sheetName in tqdm.tqdm(excelFile.sheet_names):
        tempDf = pd.read_excel(
            "data/metaphor_detection.xlsx", sheet_name=sheetName)
        keyWordSeries = pd.Series(np.array(
            [re.findall("[a-zA-Z]+", sheetName)[0] for i in range(tempDf.shape[0])]))
        tempDf["nodeWord"] = keyWordSeries
        corpusDf = pd.concat([corpusDf, tempDf])

    print("dataframe preprocessing ...")
    corpusDf.reset_index(drop=True, inplace=True)
    # corpusDf.columns = ["nodeWord", "type", "message"]
    corpusDf["message"] = corpusDf["message"].apply(lambda x: x.lower())
    corpusDf["type"] = corpusDf["type"].apply(lambda x: transTF(x))

    print("saving data ...")
    corpusDf.to_csv("data/structuredData.csv")
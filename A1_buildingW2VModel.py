#coding:utf8
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
if __name__=="__main__":
        '''
        build w2v model
        '''

        print("loading data ...")
        samplesDf=pd.read_csv("data/structuredData.csv")
        corpusList=[row.split(" ") for row in np.array(samplesDf["message"]).tolist()]

        print("training model ...")
        myW2VModel=Word2Vec(corpusList,window=3,iter=100,min_count=0)
        myW2VModel.train(corpusList,\
                total_examples=myW2VModel.corpus_count,\
                epochs=myW2VModel.epochs)

        print("saving model ...")
        myW2VModel.save("model/w2vModel.model")

        print("finished!")

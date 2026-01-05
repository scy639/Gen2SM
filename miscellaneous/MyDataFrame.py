import confs
import json
import os
import numpy as np
import torch
import pandas as pd


class MyDataFrameA:
    def __init__(self):
        self.cur_k = None
        self.k2dic = {}
    def new_k(self,new_k):
        assert new_k not in self.k2dic,(new_k,self.k2dic)
        self.k2dic[new_k]={}
        self.cur_k=new_k
    def set_cur_dic(self,k,v):
        """
        cur_dic means k2dic[cur_k], not k2dic
        """
        self.k2dic[self.cur_k][k]=v
    def get_cur_dic(self,k,):
        return self.k2dic[self.cur_k][k]
    def get_df(self):
        df=pd.DataFrame.from_dict(self.k2dic, orient='index')
        return df
    def clear(self):
        self.cur_k = None
        self.k2dic = {}
'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    entropy = 0
    target = df[[df.columns[-1]]].values
    _, counts = np.unique(target, return_counts=True)
    total_count = np.sum(counts)
    for freq in counts:
        temp = freq/total_count
        if temp != 0:
            entropy -= temp*(np.log2(temp))
    return entropy

def get_avg_info_of_attribute(df, attribute):
    attribute_value = df[attribute].value_counts().to_dict()
    average_info = 0
    
    for i in attribute_value:
    	att_df = df[df[attribute] == i]
    	average_info += (attribute_value[i]/df[attribute].shape[0])*get_entropy_of_dataset(att_df)
    
    return average_info

def get_information_gain(df, attribute):
    return get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)
 

def get_selected_attribute(df):
    columns = df.columns.to_list()
    columns.pop()
    mem = dict()
    for i in columns:
    	mem[i] = get_information_gain(df, i)
    return (mem, max(mem, key = lambda x: mem[x]))

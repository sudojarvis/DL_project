
# Data processing script for vega sample files

import os
import random
import json
from dateutil.parser import parse
import datetime
from save_and_load import save_split_data
# First,the model must select a subset of fields to focus on when creating visualizations
# the model must learn differences in data types across the data fields (numeric, string, temporal, ordinal, categorical etc.), which in turn guides how each field is specified in the generation of a visualization specification
# he appropriate transformations to apply to a field given its data type (e.g., aggregate transform does not apply to string fields)
# view-level transforms (aggregate, bin, calculate, filter, timeUnit) and field level transforms (aggregate, bin, sort, timeUnit) supported by the Vega-Lite grammar.

# a character tokenization strategy requires more units to represent a sequence and requires a large amount of hidden layers as well as parameters to model
# long term dependencies [8]. To address this issue and scaffold the
# learning process, we perform a set of transformations. First, we
# replace string and numeric field names using a short notation —
# “str” and “num” in the source sequence (dataset)


# iteratively generate a source (a single row from the dataset) and target pair (see Figure 3)
# from each example file. Each example is then sampled 50 times (50
# different data rows with the same Vega-Lite specification) resulting
# in a total of 215,000 pairs which are then used to train our model.

# from generate_field import generate_field_types

from save_and_load import save_data,load_data
from create_data_pair import generate_data_pair


def del_file():
    if os.path.isfile('target'):
        os.remove('target')
    if os.path.isfile('train'):
        os.remove('train')




def  get_data_pair():
        
        if os.path.isfile('target') and os.path.isfile('train'):
            all_source_hold,all_target_hold=load_data()
    
        else:
            all_source_hold,all_target_hold,max_source_seq_length,max_target_seq_length = generate_data_pair()
            # print('all_target_hold',all_target_hold)
            # print('-------------------------------------------------------')
            # print('all_source_hold',all_target_hold[0])
            save_data(all_source_hold,all_target_hold)


    
        return all_source_hold,all_target_hold



def split_data(all_source_hold,all_target_hold):
    # split the data into train and test
    train_source = all_source_hold[:int(len(all_source_hold) * 0.9)]
    train_target = all_target_hold[:int(len(all_target_hold) * 0.9)]

    test_source = all_source_hold[int(len(all_source_hold) * 0.9):]
    test_target = all_target_hold[int(len(all_target_hold) * 0.9):]

    return train_source,train_target,test_source,test_target




def _data_pair():
    # del_file()
    all_source_hold,all_target_hold=get_data_pair()
    train_source,train_target,test_source,test_target=split_data(all_source_hold,all_target_hold)
    save_split_data(train_source,train_target,test_source,test_target)




def generate_vocab():
     passs
  




# del_file()
                    
# _data_pair()    

generate_vocab()


                
                    
                
                    



                
                    


# _data_pair()

# print(count)
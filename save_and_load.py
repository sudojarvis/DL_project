import json

def  save_data(all_source_hold,all_target_hold):
    with open('train', 'w',encoding='utf-8') as source_file:
        for item in all_source_hold:
            source_file.write("%s" % item+'\n')
    # source_file.close()      
    
    with open('target', 'w' ,encoding='utf-8') as target_file:
        for item in all_target_hold:
            # print('item',item)
            json.dump(json.loads(item),target_file,ensure_ascii=False)
            target_file.write('\n')
    # f.close()
# save_data(all_source_hold,all_target_hold)


def  load_data():

    with open('train', 'r') as source_file:
        all_source_hold = source_file.read().split()
    # source_file.close()

    with open('target', 'r') as target_file:
        all_target_hold = target_file.read().split()
    # target_file.close()


    return all_source_hold,all_target_hold


def save_split_data(train_source,train_target,test_source,test_target):
    with open('train_source', 'w',encoding='utf-8') as source_file:
        for item in train_source:
            source_file.write("%s" % item+'\n')
    # source_file.close()      
    
    with open('train_target', 'w' ,encoding='utf-8') as target_file:
        for item in train_target:
            # print('item',item)
            json.dump(json.loads(item),target_file,ensure_ascii=False)
            target_file.write('\n')
    # f.close()
    
    with open('test_source', 'w',encoding='utf-8') as source_file:
        for item in test_source:
            source_file.write("%s" % item+'\n')
    # source_file.close()      
    
    with open('test_target', 'w' ,encoding='utf-8') as target_file:
        for item in test_target:
            # print('item',item)
            json.dump(json.loads(item),target_file,ensure_ascii=False)
            target_file.write('\n')
    # f.close()
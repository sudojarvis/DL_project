def split_data(all_source_hold,all_target_hold):
    # split the data into train and test
    train_source = all_source_hold[:int(len(all_source_hold) * 0.9)]
    train_target = all_target_hold[:int(len(all_target_hold) * 0.9)]

    test_source = all_source_hold[int(len(all_source_hold) * 0.9):]
    test_target = all_target_hold[int(len(all_target_hold) * 0.9):]

    return train_source,train_target,test_source,test_target




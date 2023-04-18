from save_and_load import load_train_data, load_test_data, load_data

from collections import Counter

counter=Counter()

x_train, y_train = load_train_data()
x_test, y_test = load_test_data()

all_source_hold, all_target_hold = load_data()


counter_character_source = Counter()
counter_character_target = Counter()



for item in all_source_hold:
    
    counter_character_source.update(item)

for item in all_target_hold:
        
        counter_character_target.update(item)

def save_vocab_source(counter_character_source):
    with open('vocab_source', 'w',encoding='utf-8') as source_file:
        for item ,count in counter_character_source.items():
            source_file.write("%s" % item+'\t'+str(count)+'\n')

def save_vocab_target(counter_character_target):
    with open('vocab_target', 'w',encoding='utf-8') as target_file:
        for item ,count in counter_character_target.items():
            target_file.write("%s" % item+'\t'+str(count)+'\n')

save_vocab_source(counter_character_source)
save_vocab_target(counter_character_target)

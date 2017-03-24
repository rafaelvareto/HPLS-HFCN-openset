import random

from auxiliar import load_txt_file

PATH = './frgcv1/'
SETNAME = 'set_4_label.txt'

def split_known_unknown_sets(complete_tuple_list, set_size=0.5):
    label_set = set()
    for (path, label) in complete_tuple_list:
        label_set.add(label)

    known_set = set(random.sample(label_set, int(set_size * len(label_set))))
    unknown_set = label_set - known_set
    
    known_tuple = [(path, label) for (path, label) in complete_tuple_list if label in known_set]
    unknown_tuple = [(path, label) for (path, label) in complete_tuple_list if label in unknown_set]
    return known_tuple, unknown_tuple

def main():
    set_list = load_txt_file(PATH + SETNAME)
    split_known_unknown_sets(set_list)

if __name__ == "__main__":
    main()
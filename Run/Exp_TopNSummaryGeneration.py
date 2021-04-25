from stanfordcorenlp import StanfordCoreNLP
from DataLoader import loader_raw
import tqdm

if __name__ == '__main__':
    train_data, val_data, test_data = loader_raw()
    nlp = StanfordCoreNLP(r'C:\Program Files (x86)\Python36\stanford-corenlp-full-2018-10-05')
    dictionary = {}
    for treat_source in [train_data, val_data, test_data]:
        for treat_sample in tqdm.tqdm(treat_source):
            for treat_sentence in treat_sample['Sentence']:
                state = nlp.pos_tag(treat_sentence)
                for sample in state:
                    if sample[-1] in dictionary.keys():
                        dictionary[sample[-1]] += 1
                    else:
                        dictionary[sample[-1]] = 1
    with open('Result.csv', 'w') as file:
        for key in dictionary.keys():
            file.write(str(key) + ',' + str(dictionary[key]) + '\n')
    # exit()

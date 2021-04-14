import json
import tqdm
from DataLoader import loader_raw
from Tools import rouge_1_calculation, rouge_2_calculation, rouge_long_calculation

if __name__ == '__main__':
    appoint_part = 'train'
    train_data, val_data, test_data = loader_raw([appoint_part])

    total_dictionary = {}
    for treat_sample in tqdm.tqdm(train_data):
        total_dictionary[treat_sample['ID']] = {}

        split_summary = treat_sample['Summary'].split(' ')
        for index, sentence in enumerate(treat_sample['Sentence']):
            split_sentence = sentence.split(' ')
            if len(split_sentence) <= 1: continue
            rouge_1_score = rouge_1_calculation(split_sentence, split_summary)
            rouge_2_score = rouge_2_calculation(split_sentence, split_summary)
            rouge_long_score = rouge_long_calculation(split_sentence, split_summary)
            total_dictionary[treat_sample['ID']][index] = rouge_1_score + rouge_2_score + rouge_long_score
        # print(total_dictionary)
        # exit()
    json.dump(total_dictionary, open('%s_rouge12l.json' % appoint_part, 'w'))

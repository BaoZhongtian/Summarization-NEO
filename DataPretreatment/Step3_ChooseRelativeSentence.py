import json
import tqdm
from DataLoader import loader_raw
from rouge_score import rouge_scorer

from Tools import rouge_1_calculation, rouge_2_calculation, rouge_long_calculation

if __name__ == '__main__':
    appoint_part = 'train'
    train_data, val_data, test_data = loader_raw([appoint_part])

    total_dictionary = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for treat_sample in tqdm.tqdm(train_data):
        total_dictionary[treat_sample['ID']] = {}

        treat_summary = treat_sample['Summary']
        for index, sentence in enumerate(treat_sample['Sentence']):
            rouge_score = scorer.score(target=treat_summary, prediction=sentence)
            rouge1_score = rouge_score['rouge1'].fmeasure
            total_dictionary[treat_sample['ID']][index] = rouge_score['rouge1'].fmeasure + rouge_score[
                'rouge2'].fmeasure + rouge_score['rougeL'].fmeasure
        # print(total_dictionary)
        # exit()
    json.dump(total_dictionary, open('%s_rouge12l.json' % appoint_part, 'w'))

from DataLoader import loader_raw
from Tools import rouge_1_calculation, rouge_2_calculation, rouge_long_calculation
import json
import tqdm

if __name__ == '__main__':
    appoint_part = 'train'
    train_data, val_data, test_data = loader_raw(appoint_part=[appoint_part])
    total_score = {}

    treat_data = None
    if appoint_part == 'train': treat_data = train_data
    if appoint_part == 'val': treat_data = val_data
    if appoint_part == 'test': treat_data = test_data
    assert treat_data is not None

    for treat_sample in tqdm.tqdm(treat_data):

        summary = treat_sample['Summary']
        sentence = treat_sample['Sentence']

        rouge_score = []
        for indexX in range(len(sentence)):
            for indexY in range(indexX + 1, len(sentence)):
                predict = sentence[indexX] + ' ' + sentence[indexY]

                rouge_score.append([indexX, indexY, rouge_long_calculation(predict.split(' '), summary.split(' '))])
        rouge_score = sorted(rouge_score, key=lambda x: x[-1], reverse=True)
        total_score[treat_sample['ID']] = rouge_score
        # break

    json.dump(total_score, open('%s-RougeLong-Candidate.json' % appoint_part, 'w'))

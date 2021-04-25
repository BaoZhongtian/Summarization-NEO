import os
from DataLoader import loader_raw
import json
import tqdm
import numpy
from rouge_score import rouge_scorer

if __name__ == '__main__':
    appoint_part = 'test'
    train_data, val_data, test_data = loader_raw(appoint_part=[appoint_part])
    total_score = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    save_path = 'C:/ProjectData/Pretreatment/Step4.1_MultiOne_Test_3Sentences/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    treat_data = None
    if appoint_part == 'train': treat_data = train_data
    if appoint_part == 'val': treat_data = val_data
    if appoint_part == 'test': treat_data = test_data
    assert treat_data is not None

    for treat_sample in tqdm.tqdm(treat_data):
        if os.path.exists(os.path.join(save_path, treat_sample['ID'] + '.csv')): continue
        file = open(os.path.join(save_path, treat_sample['ID'] + '.csv'), 'w')
        summary = treat_sample['Summary']
        sentence = treat_sample['Sentence']

        rouge_score = []
        for indexX in range(len(sentence)):
            for indexY in range(indexX + 1, len(sentence)):
                for indexZ in range(indexY + 1, len(sentence)):
                    predict = sentence[indexX] + ' ' + sentence[indexY] + ' ' + sentence[indexZ]
                    score = scorer.score(target=summary, prediction=predict)
                    rouge_score.append(
                        [indexX, indexY, indexZ, score['rouge1'].fmeasure, score['rouge2'].fmeasure,
                         score['rougeL'].fmeasure,
                         score['rouge1'].fmeasure + score['rouge2'].fmeasure + score['rougeL'].fmeasure])
        rouge_score = sorted(rouge_score, key=lambda x: x[-1], reverse=True)
        total_score[treat_sample['ID']] = rouge_score

        for indexX in range(numpy.shape(rouge_score)[0]):
            for indexY in range(numpy.shape(rouge_score)[1]):
                if indexY != 0: file.write(',')
                file.write(str(rouge_score[indexX][indexY]))
            file.write('\n')

        file.close()

    json.dump(total_score, open('%s-RougeLong-Candidate.json' % appoint_part, 'w'))

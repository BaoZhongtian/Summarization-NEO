import os
import numpy
import tqdm
from DataLoader import loader_raw
from rouge_score import rouge_scorer

if __name__ == '__main__':
    _, _, test_data = loader_raw(appoint_part=['test'])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for index in range(20):
        total_score = [0.0, 0.0, 0.0]
        load_path = 'C:/ProjectData/Epoch%04d/' % index
        for treat_sample in test_data[1:]:
            probability = numpy.genfromtxt(fname=os.path.join(load_path, treat_sample['ID'] + '.csv'), dtype=float,
                                           delimiter=',')
            probability = sorted(probability, key=lambda x: x[-1], reverse=True)
            predict1 = treat_sample['Sentence'][int(probability[0][0])] + ' ' \
                       + treat_sample['Sentence'][int(probability[0][1])]
            predict2 = treat_sample['Sentence'][int(probability[1][0])] + ' ' \
                       + treat_sample['Sentence'][int(probability[1][1])]
            predict3 = treat_sample['Sentence'][int(probability[2][0])] + ' ' \
                       + treat_sample['Sentence'][int(probability[2][1])]
            score1 = scorer.score(target=treat_sample['Summary'], prediction=predict1)
            score2 = scorer.score(target=treat_sample['Summary'], prediction=predict2)
            score3 = scorer.score(target=treat_sample['Summary'], prediction=predict3)
            print(predict2)
            print(score1)
            print(score2)
            print(score3)
            exit()
            total_score[0] += max([score1['rouge1'].fmeasure, score2['rouge1'].fmeasure, score3['rouge1'].fmeasure])
            total_score[1] += max([score1['rouge2'].fmeasure, score2['rouge2'].fmeasure, score3['rouge2'].fmeasure])
            total_score[2] += max([score1['rougeL'].fmeasure, score2['rougeL'].fmeasure, score3['rougeL'].fmeasure])
        total_score = numpy.array(total_score) / len(test_data)
        print(total_score)

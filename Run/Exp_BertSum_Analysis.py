import os
import pickle
import numpy
from DataLoader import loader_raw
from rouge_score import rouge_scorer

if __name__ == '__main__':
    load_path = 'Result/BertSum/'
    appoint_part = 'test'
    top_n_number = 2
    train_data, val_data, test_data = loader_raw(appoint_part=[appoint_part])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    if appoint_part == 'val': treat_data = val_data
    if appoint_part == 'test': treat_data = test_data

    for epoch_index in range(18):
        predict_score = pickle.load(open(os.path.join(load_path, '%s-%04d.pkl' % (appoint_part, epoch_index)), 'rb'))
        total_score = [0, 0, 0]
        for batch_index in range(1, len(predict_score)):
            batch_score = []
            for sentence_index in range(len(predict_score[batch_index])):
                batch_score.append([sentence_index, predict_score[batch_index][sentence_index][1]])
            batch_score = sorted(batch_score, key=lambda x: x[-1], reverse=True)

            print(treat_data[batch_index]['Summary'])
            print(treat_data[batch_index]['Sentence'][batch_score[0][0]])
            print(treat_data[batch_index]['Sentence'][batch_score[1][0]])
            print(treat_data[batch_index]['Sentence'][batch_score[2][0]])
            exit()
            predict_sentence = treat_data[batch_index]['Sentence'][batch_score[0][0]] + ' ' + \
                               treat_data[batch_index]['Sentence'][batch_score[1][0]] + \
                               treat_data[batch_index]['Sentence'][batch_score[2][0]]
            score = scorer.score(target=treat_data[batch_index]['Summary'], prediction=predict_sentence)
            total_score[0] += score['rouge1'].fmeasure
            total_score[1] += score['rouge2'].fmeasure
            total_score[2] += score['rougeL'].fmeasure
        print(total_score)

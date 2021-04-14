import os
import tqdm
import numpy
from DataLoader import loader_bert_sum, loader_raw
from Tools import rouge_1_calculation

if __name__ == '__main__':
    train_loader, val_loader, test_loader = loader_bert_sum(appoint_part=['test'], batch_size=1)
    _, _, test_data = loader_raw(appoint_part=['test'])
    load_path = 'C:/ProjectData/0000/'

    choose_top_n = 2
    total_rouge = 0.0
    for batchIndex, [batchText, batchMask, batchPosition, batchLabel] in tqdm.tqdm(enumerate(test_loader)):
        data = numpy.genfromtxt(fname=os.path.join(load_path, '%06d.csv' % batchIndex), dtype=float, delimiter=',')
        score = []
        for index, sample in enumerate(data):
            score.append([index, sample[1]])
        score = sorted(score, key=lambda x: x[-1], reverse=True)

        predict = ''
        for index in range(min(choose_top_n, len(test_data[batchIndex]['Sentence']))):
            predict += test_data[batchIndex]['Sentence'][index] + ' '
        label = test_data[batchIndex]['Summary']

        # rouge_result = rouge_1_calculation(predict[0:len(label)].split(' '), label.split(' '))
        rouge_result = rouge_1_calculation(predict.split(' '), label.split(' '))
        total_rouge += rouge_result
    print(total_rouge / len(test_data))

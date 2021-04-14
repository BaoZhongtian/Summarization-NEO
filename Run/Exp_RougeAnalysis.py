import os
import tqdm
import numpy
from DataLoader import loader_raw
from Tools import rouge_1_calculation, rouge_2_calculation

if __name__ == '__main__':
    load_path = 'C:/ProjectData/Candidate3/'
    _, _, test_data = loader_raw(appoint_part=['test'])
    index = 0

    score_sum = 0.0
    for sample in tqdm.tqdm(test_data):
        index += 1

        score = numpy.genfromtxt(fname=os.path.join(load_path, '%06d.csv' % index), dtype=float, delimiter=',')
        try:
            if len(score) == 0:
                print(index)
                continue
        except:
            print(score)
            print(index)
            continue
        score = numpy.reshape(score, [-1, 3])
        score = sorted(score, key=lambda x: x[-1], reverse=True)
        best_choose = score[0]

        summary_label = sample['Summary']
        summary_predict = sample['Sentence'][int(best_choose[0])] + ' ' + sample['Sentence'][int(best_choose[1])]

        score_sum += rouge_1_calculation(model=summary_predict.split(' '), reference=summary_label.split(' '))
    print(score_sum / len(test_data))

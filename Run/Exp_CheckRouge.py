import tqdm
from DataLoader import loader_raw
from Tools import rouge_1_calculation, rouge_2_calculation, rouge_long_calculation

if __name__ == '__main__':
    train_data, val_data, test_data = loader_raw(appoint_part=['test'])
    max_rouge_score = 0.0

    counter = 0
    for treat_sample in tqdm.tqdm(test_data):
        counter += 1
        summary = treat_sample['Summary']
        sentence = treat_sample['Sentence']

        rouge_score = []
        for indexX in range(len(sentence)):
            for indexY in range(indexX + 1, len(sentence)):
                predict = sentence[indexX] + ' ' + sentence[indexY]

                rouge_score.append(rouge_long_calculation(predict.split(' '), summary.split(' ')))

        try:
            max_rouge_score += max(rouge_score)
        except:
            pass
        # print(max(rouge_score))
        # exit()
    print(max_rouge_score / counter)

# 0.5493
# 0.5525
# 0.55417

# 0.3475

import numpy
import json
import tqdm

if __name__ == '__main__':
    appoint_part = 'val'
    rouge_1_score = json.load(open('%s-Rouge1-Candidate.json' % appoint_part, 'r'))
    rouge_2_score = json.load(open('%s-Rouge2-Candidate.json' % appoint_part, 'r'))

    rouge_final_score = {}
    for key in tqdm.tqdm(rouge_1_score.keys()):
        score = rouge_1_score[key]
        another_score = rouge_2_score[key]
        for treat_sample in another_score:
            for compare_index in range(len(score)):
                if score[compare_index][0] == treat_sample[0] and score[compare_index][1] == treat_sample[1]:
                    score[compare_index][-1] += treat_sample[-1]
                    break

        score = sorted(score, key=lambda x: x[-1], reverse=True)
        rouge_final_score[key] = score
    json.dump(rouge_final_score, open('%s-RougeFinal-Candidate.json' % appoint_part, 'w'))

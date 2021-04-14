import os
import tqdm
import torch
import numpy
from DataLoader import loader_raw


def select_list_generation(text_tokenized):
    total_selection = []
    for indexX in range(len(text_tokenized)):
        for indexY in range(indexX + 1, len(text_tokenized)):
            # total_selection.append([indexX, indexY])

            for indexZ in range(indexY + 1, len(text_tokenized)):
                total_selection.append([indexX, indexY, indexZ])
    return total_selection


def select_pad(text_tokenized, appoint_index):
    total_sample = []
    for indexX in range(len(appoint_index)):
        current_sample = [101]
        for indexY in range(len(appoint_index[indexX])):
            current_sample.extend(text_tokenized[appoint_index[indexX][indexY]])
        current_sample.append(102)
        total_sample.append(current_sample)

    max_length = max([len(sample) for sample in total_sample])
    pad_result = []
    for sample in total_sample:
        pad_result.append(numpy.concatenate([sample, numpy.zeros(max_length - len(sample))]))
    return torch.LongTensor(pad_result).unsqueeze(0)


CUDA_FLAG = True
BATCH_SIZE = 6

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    _, _, test_data = loader_raw(appoint_part=['test'])
    model = torch.load('C:/PythonProject/Summarization-Neo/ModelParameter/MatchSum_cnndm_bert.ckpt')
    save_path = 'C:/ProjectData/Candidate3/'

    if not os.path.exists(save_path): os.makedirs(save_path)
    if CUDA_FLAG: model.cuda()
    model.eval()

    index = 0
    for treat_sample in tqdm.tqdm(test_data):
        index += 1
        if os.path.exists(os.path.join(save_path, '%06d.csv' % index)): continue
        with open(os.path.join(save_path, '%06d.csv' % index), 'w') as write_file:
            text_tokens = numpy.concatenate([[101], treat_sample['Text_Tokens'][0:500], [102]])
            summary_tokens = numpy.concatenate([[101], treat_sample['Summary_Tokens'][0:500], [102]])

            text_id = torch.LongTensor(text_tokens).unsqueeze(0)
            summary_id = torch.LongTensor(summary_tokens).unsqueeze(0)

            if CUDA_FLAG:
                text_id = text_id.cuda()
                summary_id = summary_id.cuda()

            choose_group = select_list_generation(treat_sample['Sentence_Tokens'])
            for batch_start in range(0, len(choose_group), BATCH_SIZE):
                current_select_set = choose_group[batch_start:batch_start + BATCH_SIZE]
                candidate_id = select_pad(
                    text_tokenized=treat_sample['Sentence_Tokens'], appoint_index=current_select_set)
                if CUDA_FLAG: candidate_id = candidate_id.cuda()

                result = model(text_id, candidate_id, summary_id)
                score = result['score'].detach().cpu().numpy()

                for indexX in range(len(current_select_set)):
                    for indexY in range(len(current_select_set[indexX])):
                        write_file.write(str(current_select_set[indexX][indexY]) + ',')
                    write_file.write(str(score[0][indexX]))
                    write_file.write('\n')
            # exit()

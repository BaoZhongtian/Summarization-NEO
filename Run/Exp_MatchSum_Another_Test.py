import os
import numpy
import torch
import json
import tqdm
from model import MatchSum_AnotherModule
from DataLoader import loader_raw, __pad__

CUDA_FLAG = True
LOAD_PATH = 'C:/PythonProject/Summarization-NEO/DataPretreatment/'
BATCH_SIZE = 3

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_loader, val_loader, test_loader = loader_raw(appoint_part=['test'])
    test_rouge = json.load(open(os.path.join(LOAD_PATH, 'test-Rouge-Candidate.json'), 'r'))
    model = MatchSum_AnotherModule.from_pretrained('bert-base-uncased')
    if CUDA_FLAG: model.cuda()
    for epoch in range(30):
        checkpoint = torch.load('Result/MatchSum_Random10/Parameter-%04d.pkl' % epoch)
        model.load_state_dict(checkpoint['ModelStateDict'])
        save_path = 'C:/ProjectData/Epoch%04d/' % epoch
        if not os.path.exists(save_path): os.makedirs(save_path)

        for treat_sample in tqdm.tqdm(test_loader):
            if os.path.exists(os.path.join(save_path, treat_sample['ID'] + '.csv')): continue
            with open(os.path.join(save_path, treat_sample['ID'] + '.csv'), 'w') as file:
                treat_rouge = test_rouge[treat_sample['ID']]

                text_ids = numpy.concatenate([[101], treat_sample['Text_Tokens'][0:500], [102]])
                summary_ids = numpy.concatenate([[101], treat_sample['Summary_Tokens'][0:500], [102]])
                text_ids = torch.LongTensor(text_ids).unsqueeze(0)
                summary_ids = torch.LongTensor(summary_ids).unsqueeze(0)
                if CUDA_FLAG:
                    text_ids = text_ids.cuda()
                    summary_ids = summary_ids.cuda()
                for search_index in range(0, len(treat_rouge), BATCH_SIZE):
                    batch_index = treat_rouge[search_index:search_index + BATCH_SIZE]
                    batch_candidate = []
                    for sample_index in batch_index:
                        sample_index = sample_index[0:2]
                        candidate_sample = numpy.concatenate([[101], treat_sample['Sentence_Tokens'][sample_index[0]],
                                                              treat_sample['Sentence_Tokens'][sample_index[1]]])
                        candidate_sample = numpy.concatenate([candidate_sample[0:500], [102]])
                        batch_candidate.append(candidate_sample)
                    batch_candidate = __pad__(batch_candidate)
                    candidate_ids = torch.LongTensor(batch_candidate)
                    if CUDA_FLAG: candidate_ids = candidate_ids.cuda()

                    loss, score = model(text_ids, summary_ids, candidate_ids)
                    score = score.detach().cpu().numpy()[0]
                    for index in range(numpy.shape(batch_index)[0]):
                        file.write(
                            str(batch_index[index][0]) + ',' + str(batch_index[index][1]) + ',' + str(
                                score[index]) + '\n')
                # exit()

        # print('HERE')

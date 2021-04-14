import os
import json
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

LOAD_PATH = 'C:/PythonProject/Summarization-NEO/DataPretreatment/'


class DatasetBertSum(Dataset):
    def __init__(self, text_tokens, mask_tokens, position_tokens, label):
        self.text_tokens = text_tokens
        self.mask_tokens = mask_tokens
        self.position_tokens = position_tokens
        self.label = label

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, index):
        return self.text_tokens[index], self.mask_tokens[index], self.position_tokens[index], self.label[index]


class DatasetMatchSumAnother(Dataset):
    def __init__(self, text_tokens, summary_tokens, best_candidate_tokens, candidate_ids):
        self.text_tokens = text_tokens
        self.summary_tokens = summary_tokens
        self.best_candidate_tokens = best_candidate_tokens
        self.candidate_ids = candidate_ids
        assert len(self.text_tokens) == len(self.summary_tokens) == len(self.best_candidate_tokens) == len(
            self.candidate_ids)

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, index):
        return self.text_tokens[index], self.summary_tokens[index], self.best_candidate_tokens[index], \
               self.candidate_ids[index]


class CollateBertSum:
    def __pad__(self, data):
        padding_data = []
        max_length = max([len(sample) for sample in data])
        for sample in data:
            padding_data.append(numpy.concatenate([sample, numpy.zeros(max_length - len(sample))]))
        return padding_data

    def __call__(self, batch):
        xs = [sample[0] for sample in batch]
        xs = self.__pad__(xs)
        ys = [sample[1] for sample in batch]
        ys = self.__pad__(ys)
        zs = [sample[2] for sample in batch]
        label = [sample[3] for sample in batch]
        return torch.LongTensor(numpy.array(xs)), torch.LongTensor(numpy.array(ys)), zs, label


class CollateMatchSumAnother:
    def __call__(self, batch):
        xs = [sample[0] for sample in batch]
        print(xs)
        exit()


def loader_raw(appoint_part=None):
    if appoint_part is None:
        appoint_part = ['train', 'val', 'test']

    if 'train' in appoint_part:
        # train_data = json.load(open(os.path.join(LOAD_PATH, 'train_data.json'), 'r'))
        train_data = json.load(open(os.path.join(LOAD_PATH, 'train_data_part.json'), 'r'))
    else:
        train_data = None

    if 'val' in appoint_part:
        val_data = json.load(open(os.path.join(LOAD_PATH, 'val_data.json'), 'r'))
    else:
        val_data = None

    if 'test' in appoint_part:
        test_data = json.load(open(os.path.join(LOAD_PATH, 'test_data.json'), 'r'))
    else:
        test_data = None
    return train_data, val_data, test_data


def loader_bert_sum(appoint_part=None, top_n=5, batch_size=4):
    def choose_top_n(data, rouge):
        text_tokens, mask_tokens, position_tokens, label = [], [], [], []
        for treat_sample in data:
            if treat_sample['ID'] not in rouge:
                raise RuntimeError('ID not existed in rouge12l')
            score = []
            for key in rouge[treat_sample['ID']]:
                score.append([int(key), rouge[treat_sample['ID']][key]])
            score = sorted(score, key=lambda x: x[-1], reverse=True)

            text_tokens_sample, mask_tokens_sample, position_tokens_sample = [], [], []
            label_sample = numpy.zeros(len(treat_sample['Sentence']))

            try:
                for index in range(min(top_n, len(score))):
                    label_sample[score[index][0]] = 1
            except:
                print(score)
                exit()

            zero_flag = True
            for sample in treat_sample['Sentence_Tokens']:
                position_tokens_sample.append(len(text_tokens_sample))
                text_tokens_sample.append(101)
                text_tokens_sample.extend(sample)
                if zero_flag:
                    zero_flag = False
                    mask_tokens_sample.extend(numpy.zeros(1 + len(sample), dtype=int))
                else:
                    zero_flag = True
                    mask_tokens_sample.extend(numpy.ones(1 + len(sample), dtype=int))

            text_tokens.append(text_tokens_sample)
            mask_tokens.append(mask_tokens_sample)
            position_tokens.append(position_tokens_sample)
            label.append(label_sample)

        assert len(text_tokens) == len(mask_tokens) == len(position_tokens) == len(label)
        dataset = DatasetBertSum(text_tokens, mask_tokens, position_tokens, label)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateBertSum())
        return loader

    if appoint_part is None:
        appoint_part = ['train', 'val', 'test']

    train_data, val_data, test_data = loader_raw(appoint_part)
    if 'train' in appoint_part:
        train_rouge = json.load(open(os.path.join(LOAD_PATH, 'train_rouge12l.json'), 'r'))
        train_loader = choose_top_n(train_data, train_rouge)
    else:
        train_loader = None

    if 'val' in appoint_part:
        val_rouge = json.load(open(os.path.join(LOAD_PATH, 'val_rouge12l.json'), 'r'))
        val_loader = choose_top_n(val_data, val_rouge)
    else:
        val_loader = None

    if 'test' in appoint_part:
        test_rouge = json.load(open(os.path.join(LOAD_PATH, 'test_rouge12l.json'), 'r'))
        test_loader = choose_top_n(test_data, test_rouge)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def loader_match_sum(appoint_part=None, random_choose_flag=True, keep_top_flag=True, max_length=500):
    if appoint_part is None:
        appoint_part = ['train', 'val', 'test']

    train_data, val_data, test_data = loader_raw(appoint_part)

    if 'test' in appoint_part:
        test_rouge = json.load(open(os.path.join(LOAD_PATH, 'test-RougeFinal-Candidate.json'), 'r'))
        test_article, test_summary, test_candidate_id, test_best_candidate = [], [], [], []
        for treat_sample in test_data:
            test_candidate_id.append(numpy.reshape(test_rouge[treat_sample['ID']], [-1, 3]))
            test_article.append(numpy.concatenate([[101], treat_sample['Text_Tokens'][0:max_length], [102]]))
            test_summary.append(numpy.concatenate([[101], treat_sample['Summary_Tokens'][0:max_length], [102]]))

            sample_best_candidate = [101]
            sample_best_candidate.extend(treat_sample['Sentence_Tokens'][test_rouge[treat_sample['ID']][0][0]])
            sample_best_candidate.extend(treat_sample['Sentence_Tokens'][test_rouge[treat_sample['ID']][0][1]])
            sample_best_candidate = sample_best_candidate[0:max_length]
            sample_best_candidate.append(102)
            test_best_candidate.append(sample_best_candidate)
        assert len(test_article) == len(test_summary) == len(test_candidate_id) == len(test_best_candidate)
        dataset = DatasetMatchSumAnother(test_article, test_summary, test_best_candidate, test_candidate_id)

        loader = DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, collate_fn=CollateMatchSumAnother())
        return loader


if __name__ == '__main__':
    import tqdm

    test_loader = loader_match_sum()
    exit()
    max_length = 0
    for batchIndex, [batchText, _, _, _] in tqdm.tqdm(enumerate(test_loader)):
        print('here')
        exit()
    # 2453 3074

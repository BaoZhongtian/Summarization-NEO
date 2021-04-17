import os
import json
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

LOAD_PATH = 'C:/PythonProject/Summarization-NEO/DataPretreatment/'


def __pad__(data):
    padding_data = []
    max_length = max([len(sample) for sample in data])
    for sample in data:
        padding_data.append(numpy.concatenate([sample, numpy.zeros(max_length - len(sample))]))
    return padding_data


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
    def __init__(self, text_tokens, summary_tokens, best_candidate_tokens, candidate_ids, sentence_tokens):
        self.text_tokens = text_tokens
        self.summary_tokens = summary_tokens
        self.best_candidate_tokens = best_candidate_tokens
        self.candidate_ids = candidate_ids
        self.sentence_tokens = sentence_tokens
        assert len(self.text_tokens) == len(self.summary_tokens) == len(self.best_candidate_tokens) == len(
            self.candidate_ids) == len(self.sentence_tokens)

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, index):
        return self.text_tokens[index], self.summary_tokens[index], self.best_candidate_tokens[index], \
               self.candidate_ids[index], self.sentence_tokens[index]


class CollateBertSum:
    def __call__(self, batch):
        xs = [sample[0] for sample in batch]
        xs = __pad__(xs)
        ys = [sample[1] for sample in batch]
        ys = __pad__(ys)
        zs = [sample[2] for sample in batch]
        label = [sample[3] for sample in batch]
        return torch.LongTensor(numpy.array(xs)), torch.LongTensor(numpy.array(ys)), zs, label


class CollateMatchSumAnother:
    def __init__(self, random_choose_number=10):
        self.random_choose_number = random_choose_number

    def __call__(self, batch):
        xs = [sample[0] for sample in batch]
        ys = [sample[1] for sample in batch]
        zs = [sample[2] for sample in batch][0]
        ids = [sample[3] for sample in batch][0]
        sentence_tokens = [sample[4] for sample in batch][0]
        # print(numpy.shape(xs), numpy.shape(ys), numpy.shape(zs), numpy.shape(ids), numpy.shape(sentence_tokens))
        candidate_tokens = []

        random_choose_id = numpy.arange(1, numpy.shape(ids)[0])
        numpy.random.shuffle(random_choose_id)
        random_choose_id = sorted(random_choose_id[0:self.random_choose_number])
        candidate_tokens.append(zs)

        for index in range(len(random_choose_id)):
            candidate_sample = [101]
            candidate_sample.extend(sentence_tokens[int(ids[int(random_choose_id[index])][0])])
            candidate_sample.extend(sentence_tokens[int(ids[int(random_choose_id[index])][1])])
            candidate_sample.append(102)
            candidate_tokens.append(candidate_sample)
        candidate_tokens = __pad__(candidate_tokens)
        return torch.LongTensor(xs), torch.LongTensor(ys), torch.LongTensor(candidate_tokens)


def loader_raw(appoint_part=None):
    if appoint_part is None:
        appoint_part = ['train', 'val', 'test']

    if 'train' in appoint_part:
        # train_data = json.load(open(os.path.join(LOAD_PATH, 'train_data.json'), 'r'))
        train_data = json.load(open(os.path.join(LOAD_PATH, 'train_data_part.json'), 'r'))
    else:
        train_data = None

    if 'val' in appoint_part:
        # val_data = json.load(open(os.path.join(LOAD_PATH, 'val_data.json'), 'r'))
        val_data = json.load(open(os.path.join(LOAD_PATH, 'val_data_part.json'), 'r'))
    else:
        val_data = None

    if 'test' in appoint_part:
        # test_data = json.load(open(os.path.join(LOAD_PATH, 'test_data.json'), 'r'))
        test_data = json.load(open(os.path.join(LOAD_PATH, 'test_data_part.json'), 'r'))
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


def loader_match_sum(appoint_part=None, max_length=500, random_choose_number=7):
    def treatment(data, rouge):
        article, summary, candidate_id, best_candidate, sentence_tokens = [], [], [], [], []
        for treat_sample in data:
            if not rouge[treat_sample['ID']]: continue
            candidate_id.append(numpy.reshape(rouge[treat_sample['ID']], [-1, 3]))
            article.append(numpy.concatenate([[101], treat_sample['Text_Tokens'][0:max_length], [102]]))
            summary.append(numpy.concatenate([[101], treat_sample['Summary_Tokens'][0:max_length], [102]]))
            sentence_tokens.append(treat_sample['Sentence_Tokens'])

            sample_best_candidate = [101]
            sample_best_candidate.extend(treat_sample['Sentence_Tokens'][rouge[treat_sample['ID']][0][0]])
            sample_best_candidate.extend(treat_sample['Sentence_Tokens'][rouge[treat_sample['ID']][0][1]])
            sample_best_candidate = sample_best_candidate[0:max_length]
            sample_best_candidate.append(102)
            best_candidate.append(sample_best_candidate)
        assert len(article) == len(summary) == len(candidate_id) == len(best_candidate)
        dataset = DatasetMatchSumAnother(
            article, summary, best_candidate, candidate_id, sentence_tokens)
        loader = DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, collate_fn=CollateMatchSumAnother(random_choose_number))
        return loader

    if appoint_part is None:
        appoint_part = ['train', 'val', 'test']

    train_data, val_data, test_data = loader_raw(appoint_part)

    if 'train' in appoint_part:
        train_rouge = json.load(open(os.path.join(LOAD_PATH, 'train-Rouge-Candidate.json'), 'r'))
        train_loader = treatment(train_data, train_rouge)
    else:
        train_loader = None

    if 'val' in appoint_part:
        val_rouge = json.load(open(os.path.join(LOAD_PATH, 'val-Rouge-Candidate.json'), 'r'))
        val_loader = treatment(val_data, val_rouge)
    else:
        val_loader = None

    if 'test' in appoint_part:
        test_rouge = json.load(open(os.path.join(LOAD_PATH, 'test-Rouge-Candidate.json'), 'r'))
        test_loader = treatment(test_data, test_rouge)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import tqdm

    train_loader, val_loader, test_loader = loader_match_sum(appoint_part=['test'])
    for batchIndex, [batchTextTokens, batchSummaryTokens, batchCandidateTokens] in tqdm.tqdm(
            enumerate(test_loader)):
        print(numpy.shape(batchTextTokens), numpy.shape(batchSummaryTokens), numpy.shape(batchCandidateTokens))
        exit()
    # 2453 3074

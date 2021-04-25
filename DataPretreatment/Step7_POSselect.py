import os
import tqdm
import json
from stanfordcorenlp import StanfordCoreNLP
from DataLoader import loader_raw
from pytorch_pretrained_bert import BertTokenizer

CHOOSE_SET = ['NN', 'NNP', 'NNS', 'VBD', 'VB', 'VBN', 'VBZ', 'VBG', 'VBP', 'NNPS']

if __name__ == '__main__':
    train_data, val_data, test_data = loader_raw()
    nlp = StanfordCoreNLP(r'C:\Program Files (x86)\Python36\stanford-corenlp-full-2018-10-05')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    appoint_part = 'train'
    total_data = []

    for treat_sample in tqdm.tqdm(train_data):
        sample_data = {}
        sample_data['ID'] = treat_sample['ID']
        sample_data['Sentence'] = []
        sample_data['Sentence_Tokens'] = []
        sample_data['Summary'] = ''
        sample_data['Summary_Tokens'] = []

        state = nlp.pos_tag(treat_sample['Summary'])
        shrink_sentence = ''
        for word in state:
            if word[1] in CHOOSE_SET:
                shrink_sentence += word[0] + ' '
        sample_data['Summary'] = shrink_sentence
        sample_data['Summary_Tokens'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(shrink_sentence))

        for treat_sentence in treat_sample['Sentence']:
            state = nlp.pos_tag(treat_sentence)
            shrink_sentence = ''
            for word in state:
                if word[1] in CHOOSE_SET:
                    shrink_sentence += word[0] + ' '
            sample_data['Sentence'].append(shrink_sentence)

            if shrink_sentence == '':
                sample_data['Sentence_Tokens'].append([0])
                continue
            shrink_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(shrink_sentence))
            sample_data['Sentence_Tokens'].append(shrink_tokens)

        # print(sample_data)
        # for sample in sample_data:
        #     print(sample)
        #     print(sample_data[sample])
        # exit()

    json.dump(total_data, open('%s_data_shrink.json' % appoint_part, 'w'))

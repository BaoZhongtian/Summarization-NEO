import os
import json
import tqdm
import pytorch_pretrained_bert

if __name__ == '__main__':
    load_path = 'C:/ProjectData/Pretreatment/Step2_SeparatePart'
    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-base-uncased')

    for part_name in ['train']:
        total_dictionary = []
        for file_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))):
            sample_dictionary = {}
            with open(os.path.join(load_path, part_name, file_name), 'r', encoding='UTF-8') as file:
                data = file.read()
            data = data.split('\n\n')
            sample_dictionary['ID'] = file_name.replace('.story', '')
            sample_dictionary['Text'] = ''
            sample_dictionary['Sentence'] = []
            sample_dictionary['Summary'] = ''
            abstract_flag = False
            for sample in data:
                sample = sample.replace('\n', ' ').replace('\xa0', '')
                if sample == '@highlight':
                    abstract_flag = True
                    continue
                if not abstract_flag:
                    sample_dictionary['Text'] += sample + ' '
                    sample_dictionary['Sentence'].append(sample)
                else:
                    sample_dictionary['Summary'] += sample + ' '

            sample_dictionary['Text_Tokens'] = []
            sample_dictionary['Sentence_Tokens'] = []
            sample_dictionary['Summary_Tokens'] = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(sample_dictionary['Summary']))
            for sample in sample_dictionary['Sentence']:
                sample_dictionary['Sentence_Tokens'].append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample)))
            for sample in sample_dictionary['Sentence_Tokens']:
                sample_dictionary['Text_Tokens'].extend(sample)

            total_dictionary.append(sample_dictionary)

        json.dump(total_dictionary, open('%s_data.json' % part_name, 'w'))

import json
import os

if __name__ == '__main__':
    LOAD_PATH = 'C:/PythonProject/Summarization-NEO/DataPretreatment/'
    data = json.load(open(os.path.join(LOAD_PATH, 'train_data.json'), 'r'))
    part_data = data[0:10000]
    json.dump(part_data, open('train_data_part.json', 'w'))

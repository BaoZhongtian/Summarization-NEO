import json
import os

if __name__ == '__main__':
    LOAD_PATH = 'C:/PythonProject/Summarization-NEO/DataPretreatment/'
    data = json.load(open(os.path.join(LOAD_PATH, 'test_data.json'), 'r'))
    part_data = data[0:1000]
    json.dump(part_data, open('test_data_part.json', 'w'))

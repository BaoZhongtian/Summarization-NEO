import os
import numpy
import json
import tqdm

if __name__ == '__main__':
    load_path = 'C:/ProjectData/Pretreatment/Step4.1_MultiOne/'
    total_data = {}
    for filename in tqdm.tqdm(os.listdir(load_path)):
        data = numpy.genfromtxt(
            fname=os.path.join(load_path, filename), dtype=float, delimiter=',').reshape([-1, 6]).tolist()
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(2):
                data[indexX][indexY] = int(data[indexX][indexY])
        total_data[filename.replace('.csv', '')] = data

    json.dump(total_data, open('train-Rouge-Candidate.json', 'w'))

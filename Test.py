import json
import numpy

if __name__ == '__main__':
    data = json.load(open('DataPretreatment/test-Rouge-Candidate.json', 'r'))
    total_data = []
    for sample in data.keys():
        try:
            total_data.append(data[sample][0])
        except:
            pass
    print(numpy.shape(total_data))
    print(numpy.average(total_data, axis=0))

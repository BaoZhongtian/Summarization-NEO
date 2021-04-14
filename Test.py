import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    load_path = 'Run/Result/BertSum'
    total_data = []
    for index in range(4):
        data = numpy.genfromtxt(fname=os.path.join(load_path, 'Loss-%04d.csv' % index), dtype=float, delimiter=',')
        total_data.extend(data)

    another_data = []
    for index in range(0, len(total_data), 50):
        another_data.append(numpy.average(total_data[index:index + 50]))

    plt.xlabel('Train Batch Number')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss Function')
    plt.plot(another_data)
    plt.show()

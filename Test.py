import numpy

if __name__ == '__main__':
    data = numpy.genfromtxt(fname='Run/Result.csv', dtype=str, delimiter=',')
    data = sorted(data, key=lambda x: int(x[-1]), reverse=True)
    for indexX in range(numpy.shape(data)[0]):
        for indexY in range(numpy.shape(data)[1]):
            print(data[indexX][indexY], end='\t')
        print()
    # , , 378821

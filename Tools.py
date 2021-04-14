import numpy


def rouge_1_calculation(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    grams_reference = list(reference)
    grams_model = list(model)
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def rouge_2_calculation(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    grams_reference = list(model)
    grams_model = list(reference)
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


def rouge_long_calculation(model, reference):
    grams_model = list(model)
    grams_reference = list(reference)
    # print(len(grams_model),len(grams_reference))

    repetitive_matrix = numpy.zeros([len(grams_reference), len(grams_model)])

    for indexX in range(len(grams_reference)):
        for indexY in range(len(grams_model)):
            if grams_reference[indexX] == grams_model[indexY]:
                if indexX == 0 or indexY == 0:
                    repetitive_matrix[indexX][indexY] = 1
                else:
                    repetitive_matrix[indexX][indexY] = repetitive_matrix[indexX - 1][indexY - 1] + 1
            else:
                if indexX == 0 and indexY == 0: continue
                if indexX == 0:
                    repetitive_matrix[indexX][indexY] = repetitive_matrix[indexX][indexY - 1]
                if indexY == 0:
                    repetitive_matrix[indexX][indexY] = repetitive_matrix[indexX - 1][indexY]
                if indexX != 0 and indexY != 0:
                    repetitive_matrix[indexX][indexY] = \
                        max(repetitive_matrix[indexX - 1][indexY], repetitive_matrix[indexX][indexY - 1])
    rouge_l = repetitive_matrix[-1][-1] / len(grams_reference)
    # print(repetitive_matrix[-1][-1], len(grams_model), len(grams_reference))
    return rouge_l


if __name__ == '__main__':
    rouge_long_calculation(None, None)

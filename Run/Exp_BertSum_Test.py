import os
import tqdm
import numpy
from DataLoader import loader_bert_sum
from model import BertSumEncoder
import torch


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    CUDA_FLAG = True

    load_path = 'Result/BertSum/Parameter-0002.pkl'
    save_path = 'C:/ProjectData/0002/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_loader, val_loader, test_loader = loader_bert_sum(appoint_part=['test'], batch_size=1)
    model = BertSumEncoder.from_pretrained('bert-base-uncased')
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['ModelStateDict'])

    if CUDA_FLAG: model.cuda()
    model.eval()

    for batchIndex, [batchText, batchMask, batchPosition, batchLabel] in tqdm.tqdm(enumerate(test_loader)):
        if CUDA_FLAG:
            batchText = batchText.cuda()
            batchMask = batchMask.cuda()
        if batchText.size()[1] > 1800: continue
        # if batchText.size()[1] <= 1800: continue
        predict = model(batchText, batchMask, batchPosition).softmax(dim=-1).detach().cpu().numpy()

        with open(os.path.join(save_path, '%06d.csv' % batchIndex), 'w') as file:
            for indexX in range(numpy.shape(predict)[0]):
                for indexY in range(numpy.shape(predict)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(predict[indexX][indexY]))
                file.write('\n')

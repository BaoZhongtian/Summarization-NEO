import os
import tqdm
import numpy
from DataLoader import loader_bert_sum
from model import BertSumEncoder
import torch
import pickle

CUDA_FLAG = True

if __name__ == '__main__':
    save_path = 'Result/BertSum'
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_loader, val_loader, test_loader = loader_bert_sum(batch_size=1)
    model = BertSumEncoder.from_pretrained('bert-base-uncased')
    if CUDA_FLAG: model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1E-5)

    for episode_index in range(100):
        total_loss = 0.0
        model.train()
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, [batchText, batchMask, batchPosition, batchLabel] in enumerate(train_loader):
                if CUDA_FLAG:
                    batchText = batchText.cuda()
                    batchMask = batchMask.cuda()
                loss = model(batchText, batchMask, batchPosition, batchLabel)

                loss_value = loss.cpu().detach().numpy()
                total_loss += loss_value
                print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
                file.write(str(loss_value) + '\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        total_predict = []
        for batchIndex, [batchText, batchMask, batchPosition, batchLabel] in enumerate(val_loader):
            if CUDA_FLAG:
                batchText = batchText.cuda()
                batchMask = batchMask.cuda()
            predict = model(batchText, batchMask, batchPosition).softmax(dim=-1)
            predict = predict.cpu().detach().numpy()
            total_predict.append(predict)
        print('Val Part Evaluate Completed')
        pickle.dump(total_predict, open(os.path.join(save_path, 'Val-%04d.pkl' % episode_index), 'wb'))

        total_predict = []
        for batchIndex, [batchText, batchMask, batchPosition, batchLabel] in enumerate(test_loader):
            if CUDA_FLAG:
                batchText = batchText.cuda()
                batchMask = batchMask.cuda()
            predict = model(batchText, batchMask, batchPosition).softmax(dim=-1)
            predict = predict.cpu().detach().numpy()
            total_predict.append(predict)
        print('Test Part Evaluate Completed')
        pickle.dump(total_predict, open(os.path.join(save_path, 'Test-%04d.pkl' % episode_index), 'wb'))

        print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
        # torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
        #            f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))

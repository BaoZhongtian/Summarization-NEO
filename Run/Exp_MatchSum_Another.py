import os
import numpy
import torch
from model import MatchSum_AnotherModule
from DataLoader import loader_match_sum

CUDA_FLAG = True

if __name__ == '__main__':
    random_choose_number = 9
    save_path = 'Result/MatchSum_Random%d' % (random_choose_number + 1)
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_loader, val_loader, test_loader = loader_match_sum(random_choose_number=random_choose_number)
    model = MatchSum_AnotherModule.from_pretrained('bert-base-uncased')
    if CUDA_FLAG: model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1E-5)

    for episode_index in range(100):
        total_loss = 0.0
        model.train()
        with open(os.path.join(save_path, 'Loss-%04d.csv' % episode_index), 'w') as file:
            for batchIndex, [batchTextTokens, batchSummaryTokens, batchCandidateTokens] in enumerate(train_loader):
                if CUDA_FLAG:
                    batchTextTokens = batchTextTokens.cuda()
                    batchSummaryTokens = batchSummaryTokens.cuda()
                    batchCandidateTokens = batchCandidateTokens.cuda()
                if batchCandidateTokens.size(0) != random_choose_number + 1: continue
                loss, score = model(batchTextTokens, batchSummaryTokens, batchCandidateTokens)

                loss_value = loss.cpu().detach().numpy()
                total_loss += loss_value
                print('\rBatch %d Loss = %f' % (batchIndex, loss_value), end='')
                file.write(str(loss_value) + '\n')

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        print('\nEpisode %d Total Loss = %f' % (episode_index, total_loss))
        torch.save(obj={'ModelStateDict': model.state_dict(), 'OptimizerStateDict': optimizer.state_dict()},
                   f=os.path.join(save_path, 'Parameter-%04d.pkl' % episode_index))

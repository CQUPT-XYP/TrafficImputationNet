# coding:utf-8
import os
import torch
import argparse
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from data.traffic_dataset import TrafficDataset
from data.traffic_dataset import InitDataset
from net.SubNet1 import ConvRNNNet
from net.SubNet2 import PConvUNet
from net.ImputationNet import ImputationNet
from net.ImputationLoss import ImputationLoss
import datetime
from sklearn.metrics import median_absolute_error


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default='2', help='')
parser.add_argument('--epochs', type=int, default='100', help='')
parser.add_argument('--loss_ratio', type=int, default='30', help='')
parser.add_argument('--random_seed', type=int, default='1024', help='')
parser.add_argument('--save_path', type=str, default='./save', help='')
parser.add_argument('--learning_rate', type=float, default='1e-2', help='')
parser.add_argument('--mask_path', type=str, default='./data/dataset/mask.npy', help='')
parser.add_argument('--dataset_file_path', type=str, default='data/dataset/dataset_missing_{}_ratio.npy', help='')
parser.add_argument('--rnn_type', type=str, default='gru', help='lstm or gru')


args = parser.parse_args()

batch_size = args.batch_size
loss_ratio = args.loss_ratio
epochs = args.epochs
random_seed = args.random_seed
learning_rate = args.learning_rate
save_path = args.save_path
mask_path = args.mask_path
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

matrix_len = np.load(mask_path, allow_pickle=True).shape[0]

dataset_file_path = args.dataset_file_path.format(loss_ratio)
dataset = InitDataset(dataset_path=dataset_file_path).get_dataset()
train_dataset = TrafficDataset(dataset, stage_name="train", mask_path=mask_path)
valid_dataset = TrafficDataset(dataset, stage_name="val", mask_path=mask_path)
test_dataset = TrafficDataset(dataset, stage_name="test", mask_path=mask_path)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


subnet1 = ConvRNNNet(matrix_len, args.rnn_type)
subnet2 = PConvUNet()
net = ImputationNet(subnet1, subnet2).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
loss_function = ImputationLoss().to(device)


def test(test_loader):
    net.load_state_dict(torch.load(os.path.join(save_path, "best_model_{}.pth".format(loss_ratio)), map_location=device), False)
    net.eval()
    with torch.no_grad():
        loss_list = []
        mape_list = []
        rmse_list = []
        mae_list = []
        meae_list = []

        for block, center_data, loss_mask, target, loss_indices, mask in tqdm(test_loader):
            start_dt = datetime.datetime.now()
            block = block.to(device)
            center_data = center_data.to(device)
            loss_mask = loss_mask.to(device)
            mask = mask.to(device)
            target = target.to(device)
            pred = net(block, center_data, mask)
            end_dt = datetime.datetime.now()
            print((end_dt - start_dt))
            
            loss = loss_function(pred, target, loss_indices, loss_mask)
            loss_list.append(loss.item())
            mape, rmse, mae, meae = cal_loss(target, pred, loss_indices, loss_mask)

            mape_list.append(mape)
            rmse_list.append(rmse)
            mae_list.append(mae)
            meae_list.append(meae)
            
        loss_avg = np.average(loss_list)
        mape_num = np.average(mape_list)
        rmse_num = np.average(rmse_list)
        mae_num = np.average(mae_list)
        meae_num = np.average(meae_list)
        return loss_avg, mape_num, rmse_num, mae_num, meae_num

def evaluate(val_loader):
    net.eval()
    with torch.no_grad():
        loss_list = []
        for block, center_data, loss_mask, target, loss_indices, mask in tqdm(val_loader):
            block = block.to(device)
            center_data = center_data.to(device)
            loss_mask = loss_mask.to(device)
            mask = mask.to(device)
            target = target.to(device)
            pred = net(block, center_data, mask)
            loss = loss_function(pred, target, loss_indices, loss_mask)
            loss_list.append(loss.item())
        loss_avg = np.average(loss_list)
        return loss_avg


def cal_loss(real, pred, batch_loss_indices, loss_matrix):
    pred = pred.cpu()
    real = real.cpu()

    loss_matrix = loss_matrix.cpu()
    pred_result = pred * loss_matrix
    target_result = real * loss_matrix

    mask = ((loss_matrix != 0) & (target_result != 0) & (pred_result != 0))

    pred_result = pred_result[mask]
    target_result = target_result[mask]

    mape = metrics.mean_absolute_percentage_error(target_result, pred_result)

    rmse = np.sqrt(metrics.mean_squared_error(target_result, pred_result))
    mae =  metrics.mean_absolute_error(target_result, pred_result)
    meae = median_absolute_error(target_result, pred_result)
    return mape, rmse, mae, meae


def train(train_loader, val_loader, test_loader, loss_ratio):
    net.train()
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    best_val_loss = float("inf")

    stop_counter = 0
    decrease_count = 0
    batch_count = 0

    for epoch in range(epochs):
        # net.train()
        if stop_counter >= 6:
            # 如果连续n个epoch loss不下降，停止训练
            print("early stop")
            break

        epoch_loss_list = []
        for block, center_data, loss_mask, target, loss_indices, mask in tqdm(train_loader):
            print(block.shape, mask.shape)
            block = block.to(device)
            center_data = center_data.to(device)
            loss_mask = loss_mask.to(device)
            mask = mask.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = net(block, center_data, mask)
            loss = loss_function(pred, target, loss_indices, loss_mask)
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.item())

            batch_count += 1

        
        train_loss = np.average(epoch_loss_list)
        eval_loss = evaluate(val_loader=val_loader)


        print("train loss:{}, eval loss:{}, best loss:{}".format(train_loss, eval_loss, best_val_loss))


        # 如果loss下降
        if eval_loss < best_val_loss and abs(best_val_loss - eval_loss) > 0.01:
        # if eval_loss < best_val_loss:
            # torch.save(net.state_dict(), os.path.join(save_path, "best_model_{}.pth".format(loss_ratio)))
            best_val_loss = eval_loss
            print("best model saved")
            stop_counter = 0
            decrease_count = 0
        else:
            # 如果连续3个epoch loss不下降，减小学习率
            if decrease_count >= 2:
                scheduler.step()
                print("learing rate decay")
                decrease_count = 0
            else:
                decrease_count += 1
            stop_counter += 1
        
        if train_loss < best_val_loss * 0.8:
            print("overfit early stop")
            break
        
        torch.save(net.state_dict(), os.path.join(save_path, "best_model_{}.pth".format(loss_ratio)))


    # 在测试集上的结果
    test_loss, test_mape, test_rmse, test_mae, test_meae = test(test_loader=test_loader)
    print("loss ratio: {} --- dt: {} --- test loss:{}, test mape:{}, test_rmse:{}, test_mae:{}".format(loss_ratio, str(datetime.datetime.now()), test_loss, test_mape, test_rmse, test_mae))
    
    # result_file_path = os.path.join("./save_no_earlystop/result.txt")
    # with open(result_file_path, "a") as f:
    #     f.write("loss ratio: {} --- dt: {} --- test loss:{}, test mape:{}, test_rmse:{}, test_mae:{}".format(loss_ratio, str(datetime.datetime.now()), test_loss, test_mape, test_rmse, test_mae) + "\n")
    

if __name__ == '__main__':
    train(train_loader, val_loader, test_loader, loss_ratio)
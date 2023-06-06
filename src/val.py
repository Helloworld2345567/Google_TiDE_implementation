from parse_args import *
from dataloader import *
from models import *
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.utils.data import DataLoader

#val curves 
#after scale transform not original data 
def Dataset(args):
    print('loading the val data...')
    size=[args.lookback_len+args.pred_len,args.lookback_len,args.pred_len]
    
    if(args.dataset=='ETTh1' or args.dataset=='ETTh2'):
        ETdir = os.path.join(args.datadir, 'ETDataset', 'ETT-small')
        val_set = Dataset_ETT_hour(root_path=ETdir, flag='val',size=size,data_path=args.dataset+'.csv', features='M')

    elif(args.dataset=='ETTm1' or args.dataset=='ETTm2'):
        ETdir = os.path.join(args.datadir, 'ETDataset', 'ETT-small')
        val_set =Dataset_ETT_minute(root_path=ETdir, flag='val',size=size,data_path=args.dataset+'.csv', features='M')
    else:
        dir = os.path.join(args.datadir, args.dataset)
        val_set = Dataset_Custom(root_path=dir, flag='val',size=size,data_path=args.dataset+'.csv', features='MS')
        
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    print('Loading finished.  val data: ',len(val_set))
    
    return val_loader

args = parse_args()
val_loader = Dataset(args)
sizes = {}
for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(val_loader):
    sizes['lookback'] = (seq_y.shape[1], seq_y.shape[2])
    sizes['attr'] = (seq_x.shape[1], seq_x.shape[2])
    sizes['dynCov'] = (seq_y_mark.shape[1], seq_y_mark.shape[2])
    break

print('val dataloader: ',len(val_loader))

#加载已经训练好的model
model = TiDEModel(sizes, args)
model.cuda()
model.load_state_dict(torch.load(os.path.join(args.ckpt_path,'TiDE.pt')))
criterion = torch.nn.MSELoss(reduction='mean')
val_mse_loss = 0.
val_mae_loss = 0.
model.eval()
pred_list=[]
ans_list=[]
for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(val_loader):
    if args.cuda == 'True':
        seq_x = seq_x.float().cuda()
        seq_y = seq_y.float().cuda()
        # turn marks into float type
        seq_x_mark = seq_x_mark.float().cuda()
        seq_y_mark = seq_y_mark.float().cuda()
    if (i%args.pred_len==0):
        pred, ans = model(seq_x, seq_y, seq_x_mark, seq_y_mark)
        pred_list.append(torch.squeeze(pred).cpu().detach().numpy())
        ans_list.append(torch.squeeze(ans).cpu().detach().numpy())
        #print(pred.shape) 
        #print(ans.shape)
        # use MSE loss
        loss = criterion(pred, ans)
        # calculate the MAE loss
        mae_loss = torch.mean(torch.abs(pred - ans))
        val_mse_loss += loss.item()
        val_mae_loss += mae_loss.item()

val_mse_loss = val_mse_loss /len(val_loader)
val_mae_loss = val_mae_loss /len(val_loader)
print('val_mse_loss: {}, val_mae_loss: {}'.format(val_mse_loss*args.pred_len, val_mae_loss*args.pred_len))
#print(np.array(pred_list).shape)
pre=np.array(pred_list)
ans=np.array(ans_list)
# 将矩阵展平为二维矩阵 (B,H,N)->(H,B*N) 
pre = pre.reshape(pre.shape[0]*pre.shape[1], -1).T
ans = ans.reshape(ans.shape[0]*ans.shape[1], -1).T

print('val data.shape:',pre.shape)

lenth = 720
start = 0
print('总预测长度Lenth：',lenth)
time=np.arange(pre.shape[1])
plt.plot(time[0:lenth],pre[-1][start:start+lenth], 'r',linewidth=1, label='Prediction')
plt.plot(time[0:lenth],ans[-1][start:start+lenth], label='Actual')
plt.title('Valilation Curve')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(args.ckpt_path,('val_curve_'+str(lenth)+'.jpg')))
plt.show()
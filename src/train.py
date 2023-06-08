from dataloader import *
from parse_args import *
from models import *
from test import *
import os
import sys
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import json
import time

def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)
        
def Dataset(args):
    print('loading the training data...')
    size=[args.lookback_len,args.lookback_len,args.pred_len]
    
    if(args.dataset=='ETTh1' or args.dataset=='ETTh2'):
        ETdir = os.path.join(args.datadir, 'ETDataset', 'ETT-small')
        train_set = Dataset_ETT_hour(root_path=ETdir, flag='train',size=size,data_path=args.dataset+'.csv', features='M')#features='M'  use all columns
        test_set = Dataset_ETT_hour(root_path=ETdir, flag='test',size=size,data_path=args.dataset+'.csv', features='M')

    elif(args.dataset=='ETTm1' or args.dataset=='ETTm2'):
        ETdir = os.path.join(args.datadir, 'ETDataset', 'ETT-small')
        train_set =Dataset_ETT_minute(root_path=ETdir,flag='train',size=size,data_path=args.dataset+'.csv', features='M')
        test_set = Dataset_ETT_minute(root_path=ETdir, flag='test',size=size,data_path=args.dataset +'.csv', features='M')
        
    else:
        dir = os.path.join(args.datadir, args.dataset)
        train_set = Dataset_Custom(root_path=dir, flag='train', size=size, data_path=args.dataset +'.csv', features='MS')
        test_set = Dataset_Custom(root_path=dir, flag='test', size=size, data_path=args.dataset + '.csv', features='MS')
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size,shuffle=False,num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    print('Loading finished. Train data: {}, test data: {}'.format(len(train_set), len(test_set)))
    
    return train_loader,test_loader
        
if __name__ == '__main__':

    ############################## 1. Parse arguments ################################
    print('parsing arguments...')
    args = parse_args()
    
    # create the checkpoint directory if it does not exist
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    
    if args.print_tofile == 'True':
        # Open files for stdout and stderr redirection
        stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
        stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
        # Redirect stdout and stderr to the files
        sys.stdout = stdout_file
        sys.stderr = stderr_file
    
    save_args_to_file(args, os.path.join(args.ckpt_path, 'args.json'))
   
    # print args
    print('args:\n',args)

    ############################## 2. preprocess and loading the training data ################################
    
    train_loader,test_loader=Dataset(args)
    # get sizes
    sizes = {}
    for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
        sizes['lookback'] = (seq_y.shape[1], seq_y.shape[2])
        sizes['attr'] = (seq_x.shape[1], seq_x.shape[2])
        sizes['dynCov'] = (seq_y_mark.shape[1], seq_y_mark.shape[2])
        break
    print(sizes)
    ############################## 3. build the model ################################
    
    print('model built.')
    model = TiDEModel(sizes, args)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.cuda == 'True':
        model.cuda()
    
    
    optim = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optim, T_max=args.epoch * len(train_loader))
    criterion = torch.nn.MSELoss(reduction='mean')

    training_stats = {
        'epoch': [],
        'train_mse': [],
        'train_mae': [],
        'test_mse': [],
        'test_mae': [],
        'running_time': []
    }
    
    ############################## 4. train the model ################################
    start_time = time.time()
    best_loss = 9999
    print('batch num: {}'.format(len(train_loader)))
    for epoch in range(1, args.epoch + 1):
        
        train_mse_loss = 0.
        train_mae_loss = 0.
        test_mse_loss = 0.
        test_mae_loss = 0.
        
        step = 0
        model.train()
        #print('Starting epoch: {}'.format(epoch))
        for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
            
            step += 1
            if epoch == 1 and step <= 1:
                print('seq_x: {}, seq_y: {}, seq_x_mark: {}, seq_y_mark: {}'.format(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape))
               
            optim.zero_grad()
            if args.cuda == 'True':
                seq_x = seq_x.float().cuda()
                seq_y = seq_y.float().cuda()
                # turn marks into float type
                seq_x_mark = seq_x_mark.float().cuda()
                seq_y_mark = seq_y_mark.float().cuda()

            pred, ans = model(seq_x, seq_y, seq_x_mark, seq_y_mark)
            # use MSE loss
            loss = criterion(pred, ans)
            
            # calculate the MAE loss
            mae_loss = torch.mean(torch.abs(pred - ans))
            train_mse_loss += loss.item()
            train_mae_loss += mae_loss.item()
            loss.backward()
            optim.step()
            lr_scheduler.step()  # 在每个epoch中更新学习率 cos 1/4T ->0
            
        train_mse_loss /= len(train_loader)
        train_mae_loss /= len(train_loader)
            # test the model on test set
        test_mse_loss, test_mae_loss = test(args, model, test_loader, criterion)

        print('Epoch: {}, train_mse_loss: {:.3f}, train_mae_loss: {:.3f}, test_mse_loss: {:.3f}, test_mae_loss: {:.3f}'.format(epoch, train_mse_loss, train_mae_loss, test_mse_loss, test_mae_loss))
        if test_mse_loss < best_loss:
            best_loss = test_mse_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, '{}.pt'.format(args.name)))
            print('Best model saved.')
    
        # update training stats
        training_stats['epoch'].append(epoch)
        training_stats['train_mse'].append(train_mse_loss)
        training_stats['train_mae'].append(train_mae_loss)
        training_stats['test_mse'].append(test_mse_loss)
        training_stats['test_mae'].append(test_mae_loss)


       

    end_time = time.time()
    total_time = end_time - start_time
    print('Total running time: {} seconds'.format(total_time))
    training_stats['running_time'].append(total_time)

    np.save(os.path.join(args.ckpt_path, "training_stats.npy"), training_stats)
    
    if args.print_tofile == 'True':
        # Close the files to flush the output
        stdout_file.close()
        stderr_file.close()

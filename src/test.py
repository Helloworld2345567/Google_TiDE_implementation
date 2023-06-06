'''
The test function for TiDE.
'''

import torch

def test(args, model, val_loader, criterion):
    val_mse_loss = 0.
    val_mae_loss = 0.
    model.eval()
    for _, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(val_loader):
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
        val_mse_loss += loss.item()
        val_mae_loss += mae_loss.item()

    val_mse_loss /= len(val_loader)
    val_mae_loss /= len(val_loader)

    return val_mse_loss, val_mae_loss
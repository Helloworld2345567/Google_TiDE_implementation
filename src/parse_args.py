import argparse
#add RevIn...
def parse_args():
    parser = argparse.ArgumentParser(description='TiDE model')

    # data parameters
    parser.add_argument('--name', type=str, default='TiDE', help='name of the model')
    parser.add_argument('--print-tofile', type=str, default='True', help='print to file or not')
    parser.add_argument('--datadir', type=str, default='', help='path to the data file')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--save_path', type=str, default='', help='path to save the trained model')
    parser.add_argument('--cuda', type=str, default='True', help='use cuda or not')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset name')
    
    parser.add_argument('--lookback_len', type=int, default=720, help='lookback length L')
    parser.add_argument('--pred_len', type=int, default=24 * 4, help='prediction length H')
    parser.add_argument('--feat_size', type=int, default=2, help='size of the feature vector')
    
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to train for')

    # model parameters
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden state')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--decoder_output_dim', type=int, default=8, help='output dimension of the decoder')
    parser.add_argument('--temporal_decoder_hidden', type=int, default=128, help='hidden size of the temporal decoder')
    parser.add_argument('--layer_norm', type=str, default='True', help='use layer_norm or not')
    parser.add_argument('--revin', type=str, default='False', help='use revin or not')
    parser.add_argument('--drop_prob', type=float, default=0.3, help='dropout probability')
    parser.add_argument('--lr', type=float, default=3.82e-5, help='learning rate')

    args = parser.parse_args()

    return args
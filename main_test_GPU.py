import os
import json
import time
import utils
import models.DeepTTE
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str, default='test')
parser.add_argument('--batch_size', type = int, default = 18000)
parser.add_argument('--epochs', type = int, default = 100)

# evaluation args
parser.add_argument('--weight_file', type = str, default='saved_weight_GPU')
parser.add_argument('--result_file', type = str, default='result/result_GPU.res')

# cnn args
parser.add_argument('--kernel_size', type = int, default=3)

# rnn args
parser.add_argument('--pooling_method', type = str, default='attention')

# multi-task args
parser.add_argument('--alpha', type = float, default=0.1 )

# log file name
parser.add_argument('--log_file', type = str, default='run_log_GPU.txt')

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))


def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    final_loss = 0.0
    final_idx = 0.0
    final_deviation = np.array([])
    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result: write_result(fs, pred_dict, attr)

            running_loss += loss.data
            final_loss += loss.data

        final_idx += idx + 1.0
        print( 'Evaluate on file {}, MAPE {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, MAPE {}'.format(input_file, running_loss / (idx + 1.0)))

        label = pred_dict['label'].data.cpu().numpy()
        pred = pred_dict['pred'].data.cpu().numpy()
        deviation = np.abs(label - pred)
        if final_deviation.size == 0:
            final_deviation = deviation
        else:
            final_deviation = np.concatenate( (final_deviation,deviation) )
        print( 'Evaluate on file {}, MAE {}'.format(input_file, deviation.mean() ))
        elogger.log('Evaluate File {}, MAE {}'.format(input_file, deviation.mean() ))
        RMSE=np.sqrt((np.square(deviation)).mean())
        print( 'Evaluate on file {}, RMSE {}'.format(input_file, RMSE ))
        elogger.log('Evaluate File {}, RMSE {}'.format(input_file, RMSE ))

    print('Evaluate on Final, MAPE {}'.format(final_loss / final_idx)  )
    print('Evaluate on Final, MAE {}'.format(final_deviation.mean()) )
    RMSE = np.sqrt((np.square(final_deviation)).mean())
    print('Evaluate on Final, RMSE {}'.format(RMSE) )

    if save_result: fs.close()

def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():

    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)


    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
   run()

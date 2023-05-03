import argparse
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Models.dataloader.samplers import CategoriesSampler
import time
from datetime import datetime

from Models.models.Network import DeepEMD_CL
from Models.utils import *
from Models.dataloader.data_utils import *

# Dataset path
DATA_DIR='/root/FSL/FSL_dataset/'

parser = argparse.ArgumentParser()

# about dataset and network
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet','tieredimagenet'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
# about pre-training
parser.add_argument('-max_epoch', type=int, default=120)
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-step_size', type=int, default=30)
parser.add_argument('-gamma', type=float, default=0.2)
parser.add_argument('-bs', type=int, default=128)
# about validation
parser.add_argument('-set', type=str, default='val', choices=['val', 'test'], help='the set for validation')
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15)
parser.add_argument('-metric', type=str, default='cosine')
parser.add_argument('-num_episode', type=int, default=2000)
parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
parser.add_argument('-random_val_task', action='store_true', help='random samples tasks for validation in each epoch')
# about deepemd setting
parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
parser.add_argument('-feature_pyramid', type=str, default=None)
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# about training
parser.add_argument('-gpu', default='0')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')

# Contrastive Settings
parser.add_argument('-con_momentum', type=float, default=0.99)
parser.add_argument('-temparature', type=float, default=0.07)
parser.add_argument('-mlp_dim', type=int, default=128)
parser.add_argument('-alpha', type=float, default=1.0)
parser.add_argument('-queue_len', type=int, default=32768)

parser.add_argument('-train_time', default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), type=str, metavar='PATH', help='path to cache (default: none)')

args = parser.parse_args()
pprint(vars(args))

num_gpu = set_gpu(args)
set_seed(args.seed)

dataset_name = args.dataset
args.save_path = 'pre_train/%s/%d-%.4f-%d-%.2f/' % \
                 (dataset_name, args.bs, args.lr, args.step_size, args.gamma)
args.save_path = osp.join('checkpoint', args.save_path, args.train_time)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)

Dataset=set_up_datasets(args)
trainset = Dataset('train', args)
train_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=32, pin_memory=True, drop_last=False)

valset = Dataset('val', args)
val_sampler = CategoriesSampler(valset.label, args.num_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=32, pin_memory=True)
print('save all checkpoint models:', (args.save_all is True))

model = DeepEMD_CL(args, mode='pre_train')
model = model.cuda()

# label of query images.
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)  # shape[75]:012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()

optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': args.lr}
                             ], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


def save_model(name):
    torch.save(dict(params=model.encoder.state_dict()), osp.join(args.save_path, name + '.pth'))


trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path, 'tf'))

result_list = [args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print(args.save_path)
    start_time = time.time()
    optimizer.zero_grad()
        
    tl1 = Averager()
    tl = Averager()
    a_c1 = Averager()
    a_c = Averager()
    a_r = Averager()
    tqdm_gen = tqdm.tqdm(train_loader)
    for i, batch in enumerate(tqdm_gen, 1):
        model = model.train()
        global_count = global_count + 1
        data, rot_images, train_label = [_.cuda() for _ in batch]

        # model + SSL
        #########################################
        # Classical classification Loss
        #########################################
        model.mode = 'pre_train'
        logits_cls = model(data)
        loss_cls = F.cross_entropy(logits_cls, train_label)

        #########################################
        # Rotation Loss
        #########################################
        model.mode = 'rot'
        loss_rot, rot_pred, rot_label = model(rot_images)

        # acc compute
        acc_cls = count_acc(logits_cls, train_label)
        acc_rot = count_acc(rot_pred, rot_label)

        # Loss & backword
        total_loss = args.alpha * loss_cls + (1.0 - args.alpha) * loss_rot

        tl.add(total_loss.item())
        a_c.add(acc_cls)
        a_r.add(acc_rot)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Total Loss: ', tl.item())
    print('Classical Acc: ', a_c.item())
    print('Rotation Acc: ', a_r.item())

    lr_scheduler.step()
    model = model.eval()
    model.mode = 'meta'

    vl = Averager()
    va = Averager()
    vl1 = Averager()
    va1 = Averager()

    #use deepemd fcn for validation
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data_ori, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot

            # encoder data by encoder
            model.mode = 'encoder'
            data = model(data_ori)
            data_shot, data_query = data[:k], data[k:]

            # episode learning
            model.mode = 'meta'
            if args.shot > 1:   # k-shot case
                data_shot = model.get_sfc(data_shot)
            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))    # repeat for multi-gpu processing
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

    print('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    if va >= trlog['max_acc']:
        print('A better model is found!!')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')
        torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_best.pth'))

    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('This epoch takes %d seconds' % (time.time() - start_time),
          '\nstill need around %.2f hour to finish' % ((time.time() - start_time) * (args.max_epoch - epoch) / 3600))
    

writer.close()
result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ))
save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

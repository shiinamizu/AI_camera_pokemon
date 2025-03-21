from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from datasets.pokemon import Pokemon
from learn.model import VIT as Models

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    

    header = 'Epoch: [{}]'.format(epoch)
    # for clip, target, video_len,text_len in metric_logger.log_every(data_loader, print_freq, header): 
    for img, target, video_idx in data_loader: 
        start_time = time.time()
        img, target = img.to(device), target.to(device)
        output = model(img)
        loss = criterion(output, target)    #nn.CrossEntropyLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                                      

        batch_size = img.shape[0]
        
        lr_scheduler.step()
        sys.stdout.flush()
    return loss

def evaluate(model, criterion, data_loader, device,epoch_n):
    model.eval()
    
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, target, video_idx in data_loader:
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            prob = F.softmax(input=output, dim=1)

            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    result2d =np.zeros((data_loader.dataset.num_classes, data_loader.dataset.num_classes))

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
        result2d[label][v]=result2d[label][v]+1
    
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]
    print(class_count)
    print(class_correct)
    print(result2d)
    print(' * Video Acc@1 %f'%total_acc)
    print(' * Class Acc@1 %s'%str(class_acc))

    return total_acc


def main(args):

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = Pokemon(
            root=args.data_path,
            train=True
    )

    dataset_test = Pokemon(
            root=args.data_path,
            train=False
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model()

    # print(torch.cuda.is_available())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start training")
    start_time = time.time()
    acc = 0
    result =[]
    all_loss=[]

    for epoch in range(args.start_epoch, args.epochs):
        loss =train_one_epoch(model, criterion, optimizer,data_loader, device, epoch, args.print_freq)
        ev =evaluate(model, criterion, data_loader_test, device=device,epoch_n=epoch)
        acc = max(acc, ev)
        result.append(ev)
        loss = loss.to('cpu').detach().numpy().copy()
        all_loss.append(loss.item())

       


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--data-path', default='datasets', type=str, help='dataset')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='lr')
    
    # parser.add_argument('--test', default=False, type=bool, metavar='N', help='start epoch')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


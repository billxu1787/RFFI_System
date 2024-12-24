from torch import nn
import torch
import copy
import random
import numpy as np
import datetime
from torch.utils.data import DataLoader
from dataset_utils import UnsupervisedDataset, SupervisedDataset

import time
from tqdm import tqdm

from torch.optim import Adam, RMSprop, SGD
from utils import LRScheduler, EarlyStopping
import os  # Import the os module
import psutil  # Import psutil library for monitoring CPU and memory usage


def client_train_supervised(args, model, data_train, data_valid, label_train, label_valid, FineTune=False):
    model.to(args.device)

    num_class = label_train.shape[1]

    metric_fc = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_class)
    )

    metric_fc.to(args.device)

    dataset_train = SupervisedDataset(data_train, label_train)
    dataset_valid = SupervisedDataset(data_valid, label_valid)

    train_generator = DataLoader(dataset_train,
                                 batch_size=args.B,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

    valid_generator = DataLoader(dataset_valid,
                                 batch_size=args.B,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

    # Training parameters
    epochs = 1000
    min_valid_loss = np.inf

    criterion = nn.CrossEntropyLoss()

    if FineTune:
        optimizer = Adam([{'params': model.parameters(), 'lr': 1e-3},
                          {'params': metric_fc.parameters(), 'lr': 1e-3}])
    else:
        optimizer = Adam([{'params': model.parameters(), 'lr': 3e-4},
                          {'params': metric_fc.parameters(), 'lr': 3e-4}])

    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    training_loss, training_acc = [], []
    valid_loss, valid_acc = [], []

    for epoch in range(epochs):

        model.train()
        training_running_loss = 0.0
        training_running_correct = 0

        for iteration, (inputs, labels) in enumerate(train_generator):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()

            feature = model(inputs)
            outputs = metric_fc(feature)

            preds = torch.argmax(outputs, dim=1)
            truth = torch.argmax(labels, dim=1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_running_loss += loss.item()
            training_running_correct += (preds == truth).sum().item()

        training_epoch_loss = training_running_loss / (iteration + 1)
        training_epoch_acc = 100.0 * training_running_correct / ((iteration + 1) * args.B)

        model.eval()
        valid_running_loss = 0.0
        valid_running_correct = 0

        for iteration, (inputs, labels) in enumerate(valid_generator):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            feature_v = model(inputs)
            outputs_v = metric_fc(feature_v)

            loss = criterion(outputs_v, labels)

            valid_running_loss += loss.item()
            valid_running_correct += (preds == truth).sum().item()

        valid_epoch_loss = valid_running_loss / (iteration + 1)
        valid_epoch_acc = 100.0 * valid_running_correct / ((iteration + 1) * args.B)

        print(datetime.datetime.today())

        print('Epoch ' + str(epoch + 1)
              + '\t Training Loss: ' + str(round(training_epoch_loss, 2))
              + '\t Validation Loss: ' + str(round(valid_epoch_loss, 2))
              + '\t Training Acc: ' + str(round(training_epoch_acc, 2))
              + '\t Validation Acc: ' + str(round(valid_epoch_acc, 2)))

        training_loss.append(training_epoch_loss)
        training_acc.append(training_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)

        if (epoch + 1) % 5 == 0:
            print("GPU Memory Usage:")
            print("Allocated:", torch.cuda.memory_allocated())
            print("Max Allocated:", torch.cuda.max_memory_allocated())

        best_model = copy.deepcopy(model)
        best_clf = copy.deepcopy(metric_fc)

        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            break

    print('Finished Training')

    return best_model, best_clf


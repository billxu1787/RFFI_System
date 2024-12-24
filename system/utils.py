import argparse
import torch
import numpy as np
import pandas as pd
import random
from torch.autograd import Variable
import h5py
from scipy import signal
import math
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from pyphysim.channels.fading import COST259_TUx, COST259_RAx, TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator, RayleighSampleGenerator


class LoadDataset():
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]
        return data_complex

    def _load_file(self, file_name, downsampling):
        data = []
        label = []
        cfo = []

        f = h5py.File(file_name, 'r')
        label_temp = f[self.labelset_name][:]
        label_temp = np.transpose(label_temp)

        data_temp = f[self.dataset_name]
        data_temp = self._convert_to_complex(data_temp)

        if downsampling:
            sig_len = data_temp.shape[1]
            data_temp = data_temp[:, 0:sig_len:2]

        cfo_temp = f['CFO'][:]
        cfo_temp = cfo_temp.flatten()  # 将 cfo_temp 转换为形状 (2000,)

        f.close()

        return data_temp, label_temp, cfo_temp

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.

        INPUT:
            FILE_PATH is the dataset path.

            DEV_RANGE specifies the loaded device range.

            PKT_RANGE specifies the loaded packets range.

        RETURN:
            DATA is the laoded complex IQ samples.

            LABLE is the true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        # if len(f.keys()) == 4:
        # snr = f['SNR'][:]
        # snr = np.transpose(snr)
        # elif len(f.keys()) == 3:
        #     print('Warning: SNR information is not included in the dataset.')
        #     snr = np.ones((1,len(label)))*(-999)
        #     snr = np.transpose(snr)

        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = len(dev_range)#label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)

        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[
                0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)

        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]
        # snr = snr[sample_index_list]

        f.close()
        return data, label

    def load_iq_samples_range(self, file_path, pkt_range, downsampling=True, cfo_comp=False):
        '''
        Load IQ samples from a dataset.

        INPUT:
            FILE_PATH is the dataset path.

            DEV_RANGE specifies the loaded device range.

            PKT_RANGE specifies the loaded packets range.

        RETURN:
            DATA is the laoded complex IQ samples.

            LABLE is the true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = np.transpose(label)

        sample_index_list = []

        for dev_idx in np.unique(label):
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)

        data = f[self.dataset_name][:]
        data = data[sample_index_list]
        data = self._convert_to_complex(data)

        cfo = f['CFO'][:]
        cfo = np.transpose(cfo)

        if downsampling:
            sig_len = data.shape[1]
            data = data[:, 0:sig_len:2]

        label = label[sample_index_list]

        f.close()

        if cfo_comp:
            Ts = 1 / 1e6
            for i in range(len(data)):
                data[i] = cfo_compensation(data[i], cfo[i], Ts)

        return data, label

    def load_multiple_rx_data(self, file_list, tx_range, pkt_range):

        num_rx = len(file_list)
        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        data = []
        tx_label = []
        rx_label = []

        for file_idx in range(num_rx):
            print('Start loading dataset ' + str(file_idx + 1))
            filename = file_list[file_idx]
            [data_temp, tx_label_temp, _] = self.load_iq_samples(filename, tx_range, pkt_range)
            rx_label_temp = np.ones(num_pkt * num_tx) * file_idx

            data.extend(data_temp)
            tx_label.extend(tx_label_temp)
            rx_label.extend(rx_label_temp)

        data = np.array(data)
        tx_label = np.array(tx_label)
        rx_label = np.array(rx_label)

        return data, tx_label, rx_label


def cfo_compensation(data, cfo, Ts):
    n = np.arange(1, len(data) + 1)
    data_shift = data * np.exp(-1j * 2 * np.pi * cfo * n * Ts)
    return data_shift


class LRScheduler:

    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:

    def __init__(self, patience=20, min_delta=0):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def plt_cm(true_label, pred_label, num_predictable_classes):
    conf_mat = confusion_matrix(true_label, pred_label)
    classes = range(1, num_predictable_classes + 1)

    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.show(block=True)


def to_onehot(label_in):
    u, label_int = np.unique(label_in, return_inverse=True)
    label_int = label_in.reshape(-1)
    #num_classes = len(u)
    num_classes = np.max(label_int) + 1
    label_one_hot = np.eye(num_classes, dtype='uint8')[label_int]

    return label_one_hot, num_classes
#def to_onehot(label_in):
#    label_in = np.squeeze(label_in)  # Ensure labels are in 1D
#    u, label_int = np.unique(label_in, return_inverse=True)
#    num_classes = len(u)
#    label_one_hot = np.eye(num_classes, dtype='uint8')[label_int]
#    # Create a mapping dictionary from one-hot to original labels
#    one_hot_to_label = {tuple(np.eye(num_classes, dtype='uint8')[i]): label for i, label in enumerate(u)}
#    return label_one_hot, num_classes, one_hot_to_label





class IQSampleInput():
    def __init__(self, ):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm

    def iq_samples(self, data):
        data = self._normalization(data)

        num_pkt = data.shape[0]
        len_pkt = data.shape[1]
        data_iq = np.empty([num_pkt, 2, len_pkt, 1])

        for i in range(num_pkt):
            data_iq[i, 0, :, 0] = np.real(data[i])
            data_iq[i, 1, :, 0] = np.imag(data[i])

        return data_iq
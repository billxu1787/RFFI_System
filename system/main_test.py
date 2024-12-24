import h5py
import numpy as np

from models import ConvNet
import matplotlib.pyplot as plt
import numpy as np
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, RMSprop, SGD

from utils import LoadDataset, to_onehot, LRScheduler, EarlyStopping, plt_cm
from dataset_utils import awgn, ChannelIndSpectrogram
from merge import merge_h5_files

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist  # 计算距离

import matplotlib

matplotlib.use('TkAgg')


class DistMeasure:
    def __init__(self, data_support_in, label_support_in, data_query_in, label_query_in, cnn_name_in, threshold=0.5):
        self.label_support, self.num_classes = to_onehot(label_support_in)
        self.label_query, _ = to_onehot(label_query_in)
        self.label_support = self.label_support.argmax(axis=-1)
        self.label_query = self.label_query.argmax(axis=-1)
        self.data_support = data_support_in
        self.data_query = data_query_in
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", self.device, "device")
        self.model = ConvNet()
        self.model.load_state_dict(torch.load(cnn_name_in))
        self.model.to(self.device)
        self.model.eval()
        self.ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        self.threshold = threshold

    def _extract_feature(self, data_test, add_noise=False):
        if len(data_test.shape) == 1:
            data_test = data_test.reshape([1, len(data_test)])
        model_out = []
        for i in range(data_test.shape[0]):
            test_input = data_test[i:i + 1]
            if add_noise:
                test_input = awgn(test_input, [25])
            test_input = self.ChannelIndSpectrogramObj.channel_ind_spectrogram(test_input)
            test_input = torch.from_numpy(test_input.astype(np.float32))
            test_input = test_input.to(self.device)
            test_input = Variable(test_input)
            start = time.time()
            test_output = self.model(test_input)
            end = time.time()
            inference_time = end - start
            test_output = test_output.cpu()
            test_output = test_output.detach().numpy()
            model_out.extend(test_output)
        model_out = np.array(model_out)
        return model_out

    def build_database(self):
        print('Start building RFF database.')
        feature_enrol = self._extract_feature(self.data_support)
        self.database_features = feature_enrol
        self.database_labels = self.label_support
        print('RFF database is built')

    def dist_measure_test(self, snr, calculate_fnr_fpr=False):
        feature_query = self._extract_feature(self.data_query)
        distances = cdist(feature_query, self.database_features, metric='cosine')
        pred_labels = []
        for i in range(len(feature_query)):
            min_distance = np.min(distances[i])
            min_distance_index = np.argmin(distances[i])
            print(f"Query sample {i}: min_distance = {min_distance}, min_distance_index = {min_distance_index}")
            if min_distance < self.threshold:
                pred_labels.append(self.database_labels[min_distance_index])
            else:
                pred_labels.append(-1)  # Mark as unknown device
        print(f"Predicted labels: {pred_labels}")
        print(f"True labels: {self.label_query}")

        # Extract unique labels from the query set
        query_labels = np.array(self.label_query)
        #print("Query labels:", query_labels)  # Debugging line
        # Create a set of labels that are present in the support set
        valid_labels = set(self.label_support)
        #print("Valid labels:", valid_labels)  # Debugging line
        # Modify query labels to be -1 if they do not appear in the support set
        modified_labels_query = np.where(np.isin(query_labels, list(valid_labels)), query_labels, -1)
        #print("Modified labels query:", modified_labels_query)  # Debugging line
        # Calculate accuracy using modified query labels
        acc = accuracy_score(modified_labels_query, pred_labels)
        print('Accuracy on query set is ' + str(acc))

        #acc = accuracy_score(self.label_query, pred_labels)
        #print('Accuracy on query set is ' + str(acc))

        cm = confusion_matrix(self.label_query, pred_labels, labels=np.arange(self.num_classes).tolist() + [-1])
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.arange(self.num_classes).tolist() + [-1]))
        plt.xticks(tick_marks, np.arange(self.num_classes).tolist() + ['Unknown'])
        plt.yticks(tick_marks, np.arange(self.num_classes).tolist() + ['Unknown'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        return acc


class DeviceDisconnect:
    def __init__(self, file_path):
        self.file_path = file_path

    def remove_label(self, label_to_remove):
        with h5py.File(self.file_path, 'r+') as f:
            data = f['data'][:]
            labels = f['label'][0]
            cfo = f['CFO'][0]
            keep_indices = np.where(labels != label_to_remove)[0]
            new_data = data[keep_indices]
            new_labels = labels[keep_indices]
            new_cfo = cfo[keep_indices]
            del f['data']
            del f['label']
            del f['CFO']
            f.create_dataset('data', data=new_data)
            f.create_dataset('label', data=new_labels.reshape(1, -1))
            f.create_dataset('CFO', data=new_cfo.reshape(1, -1))
        print(f"Data with label {label_to_remove} has been removed.")

'''
def determine_ranges(file_path):
    with h5py.File(file_path, 'r') as f:
        labels = f['label'][:]
        unique_devices = np.unique(labels).astype(int)
        num_devices = len(unique_devices)
        dev_range = range(num_devices)  # Outputs range(0, number of unique devices)

        # Assuming the number of packets per device is the same
        # Calculate number of packets per device by dividing total packets by number of devices
        total_packets = f['data'].shape[0]
        num_packets_per_device = total_packets // num_devices
        pkt_range = range(num_packets_per_device)

    return dev_range, pkt_range
'''
def determine_ranges(file_path):
    with h5py.File(file_path, 'r') as f:
        # 读取标签数据集
        labels = f['label'][:]
        # 找出所有唯一的标签，转换为整数，并减去1
        unique_devices = np.unique(labels).astype(int) - 1
    # 返回调整后的设备索引
    return unique_devices
def device_identify(file_path_support, file_path_query, threshold=0.002, snr=60):
    cnn_name = 'temp.pth'
    dev_range = determine_ranges(file_path_support)
    LoadDatasetObj = LoadDataset()
    data_support, label_support = LoadDatasetObj.load_iq_samples(file_path_support,
                                                                 dev_range=dev_range,
                                                                 pkt_range=range(900, 935))
    dev_range = determine_ranges(file_path_query)
    data_query, label_query = LoadDatasetObj.load_iq_samples(file_path_query,
                                                             dev_range=dev_range,
                                                             pkt_range=range(100, 300))
    dist_measure = DistMeasure(data_support, label_support, data_query, label_query, cnn_name, threshold)
    dist_measure.build_database()
    acc = dist_measure.dist_measure_test(snr)
    return acc


def device_access(output_file, h5_files, file_path_query, threshold=0.002, snr=60):
    cnn_name = 'temp.pth'
    selected_file_path = []
    selected_file_path.append(output_file)
    selected_file_path.append(h5_files)

    merge_h5_files(selected_file_path, output_file)

    dev_range = determine_ranges(output_file)
    LoadDatasetObj = LoadDataset()
    data_support, label_support = LoadDatasetObj.load_iq_samples(output_file,
                                                                 dev_range=dev_range,
                                                                 pkt_range=range(900, 935))
    dev_range = determine_ranges(file_path_query)
    data_query, label_query = LoadDatasetObj.load_iq_samples(file_path_query,
                                                             dev_range=dev_range,
                                                             pkt_range=range(100, 300))
    dist_measure = DistMeasure(data_support, label_support, data_query, label_query, cnn_name, threshold)
    dist_measure.build_database()
    acc = dist_measure.dist_measure_test(snr)
    return acc


def device_disconnect(file_path, label_to_remove, file_path_query, threshold=0.002, snr=60):
    cnn_name = 'temp.pth'
    disconnect_obj = DeviceDisconnect(file_path)
    disconnect_obj.remove_label(label_to_remove)

    # Re-load the support set from the updated file_path
    dev_range = determine_ranges(file_path)
    LoadDatasetObj = LoadDataset()
    data_support, label_support = LoadDatasetObj.load_iq_samples(file_path,
                                                                 dev_range=dev_range,
                                                                 pkt_range=range(900, 935))
    dev_range = determine_ranges(file_path_query)
    data_query, label_query = LoadDatasetObj.load_iq_samples(file_path_query,
                                                             dev_range=dev_range,
                                                             pkt_range=range(100, 300))
    dist_measure = DistMeasure(data_support, label_support, data_query, label_query, cnn_name, threshold)
    dist_measure.build_database()
    acc = dist_measure.dist_measure_test(snr, calculate_fnr_fpr=True)
    return acc


if __name__ == '__main__':
    print("请选择要运行的功能:")
    print("1. 设备识别")
    print("2. 设备断开")
    print("3. 设备接入")
    choice = input("请输入功能编号 (1/2/3): ")

    if choice == '1':
        file_path_support = './dataset/3merged_lora_receive_train.h5'
        file_path_query = './dataset/4merged_lora_receive_test.h5'
        #cnn_name = 'temp.pth'
        acc_identify = device_identify(file_path_support, file_path_query)
        print(f"Device Identification Accuracy: {acc_identify}")

    elif choice == '2':
        file_path = './dataset/4merged_lora_receive_train.h5'
        label_to_remove = 4
        file_path_query = './dataset/4merged_lora_receive_test.h5'
        #cnn_name = 'temp.pth'
        acc_disconnect = device_disconnect(file_path, label_to_remove, file_path_query)
        print(f"Device Disconnect Accuracy: {acc_disconnect}")

    elif choice == '3':
        # 定义文件路径
        output_file = './dataset/4merged_lora_receive_train.h5'
        h5_files = './dataset/lora_receive_5_train.h5'
        file_path_query = './dataset/4merged_lora_receive_test.h5'
        #cnn_name = 'temp.pth'
        acc_access = device_access(output_file, h5_files, file_path_query)
        print(f"Device Access Accuracy: {acc_access}")

    else:
        print("输入无效，请输入正确的功能编号 (1/2/3).")
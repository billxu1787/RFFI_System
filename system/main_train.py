import h5py
import numpy as np
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from utils import LoadDataset, to_onehot, LRScheduler, EarlyStopping
from dataset_utils import awgn, ChannelIndSpectrogram
from models import ConvNet


def data_generator(data_in, label_in, snr_range, channel_obj, device, batch_size=32):
    while True:
        sample_ind = random.sample(range(len(data_in)), batch_size)
        data_batch = data_in[sample_ind]
        data_batch = awgn(data_batch, snr_range)
        data_batch = channel_obj.channel_ind_spectrogram(data_batch)
        label_batch = label_in[sample_ind]
        data_batch = torch.from_numpy(data_batch.astype(np.float32)).to(device)
        label_batch = torch.from_numpy(label_batch.astype(np.float32)).to(device)
        yield Variable(data_batch), Variable(label_batch)


def train_model(model, classifier, train_generator, valid_generator, train_size, valid_size, batch_size, epochs=1000,
                lr=0.001, device=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([{'params': model.parameters()}, {'params': classifier.parameters()}], lr=lr)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    training_loss, valid_loss = [], []

    num_train_batches = train_size // batch_size
    num_valid_batches = valid_size // batch_size

    for epoch in range(epochs):
        model.train()
        classifier.train()
        training_running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_generator):
            if i >= num_train_batches:
                break
            optimizer.zero_grad()
            feature = model(inputs)
            outputs = classifier(feature)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_running_loss += loss.item()

        training_epoch_loss = training_running_loss / num_train_batches
        model.eval()
        classifier.eval()
        valid_running_loss = 0.0
        for i, (inputs, labels) in enumerate(valid_generator):
            if i >= num_valid_batches:
                break
            feature_v = model(inputs)
            outputs_v = classifier(feature_v)
            loss = criterion(outputs_v, labels)
            valid_running_loss += loss.item()

        valid_epoch_loss = valid_running_loss / num_valid_batches
        print(f'Epoch {epoch + 1}\t Training Loss: {training_epoch_loss}\t Validation Loss: {valid_epoch_loss}')
        training_loss.append(training_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            break

    print('Finished Training')
    return model


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



def main(file_path='./dataset/3merged_lora_receive_train.h5', batch_size=32):
    dev_range, pkt_range = determine_ranges(file_path)
    print(f"Device Range: {dev_range}")
    print(f"Packet Range: {pkt_range}")

    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(file_path, dev_range=dev_range, pkt_range=pkt_range)
    label_one_hot, num_classes = to_onehot(label_train)
    data_train, data_valid, label_train, label_valid = train_test_split(data_train, label_one_hot, test_size=0.1,
                                                                        shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    snr_range = range(80)

    train_gen = data_generator(data_train, label_train, snr_range, ChannelIndSpectrogramObj, device, batch_size)
    valid_gen = data_generator(data_valid, label_valid, snr_range, ChannelIndSpectrogramObj, device, batch_size)

    model = ConvNet().to(device)
    classifier = nn.Linear(128, num_classes).to(device)

    trained_model = train_model(model, classifier, train_gen, valid_gen, len(data_train), len(data_valid), batch_size,
                                epochs=1000, lr=0.001, device=device)

    torch.save(trained_model.state_dict(), "temp.pth")


if __name__ == '__main__':
    main()

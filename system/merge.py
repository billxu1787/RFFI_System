import h5py
import numpy as np
import os

def merge_h5_files(h5_files, output_file):
    # 加载所有文件的数据集到列表中
    cfos = []
    datas = []
    labels = []

    for h5_file in h5_files:
        if os.path.exists(h5_file):
            with h5py.File(h5_file, 'r') as f:
                # 检查数据集是否存在
                if 'CFO' in f and 'data' in f and 'label' in f:
                    cfos.append(f['CFO'][:])
                    datas.append(f['data'][:])
                    labels.append(f['label'][:].flatten())  # 展平label为一维数组
                else:
                    print(f"One or more datasets are missing in file: {h5_file}")
        else:
            print(f"File does not exist: {h5_file}")

    # 确保至少有一个文件被成功加载
    if not (cfos and datas and labels):
        print("No data to merge. Please check the files and datasets.")
        return

    # 合并数据集
    stacked_cfos = np.hstack(cfos)  # 合并所有CFO数据，形状为 (1, total_samples)
    stacked_datas = np.vstack(datas)  # 合并所有data数据，形状为 (total_samples, feature_size)
    stacked_labels = np.hstack(labels)  # 合并所有label数据，形状为 (total_samples,)

    # 确保合并后的数据形状一致
    assert stacked_cfos.shape[1] == stacked_datas.shape[0], "The number of samples in 'CFO' and 'data' should match."
    assert stacked_labels.shape[0] == stacked_datas.shape[0], "The number of labels should match the number of samples in 'data'."

    # 排序数据：首先根据标签排序，其次保持相同标签的原始顺序
    sorted_indices = np.argsort(stacked_labels, kind='stable')  # 使用 'stable' 保持相同标签的原始顺序
    sorted_cfos = stacked_cfos[:, sorted_indices]  # 按照排序索引排序CFO数据
    sorted_datas = stacked_datas[sorted_indices]  # 按照排序索引排序data数据
    sorted_labels = stacked_labels[sorted_indices]  # 按照排序索引排序label数据

    # 保存到新的HDF5文件中
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('CFO', data=sorted_cfos)
        f.create_dataset('data', data=sorted_datas)
        f.create_dataset('label', data=sorted_labels.reshape(1, -1))  # 形状为 (1, total_samples)

    # 检查输出文件的内容
    with h5py.File(output_file, 'r') as f:
        print(f"CFO shape: {f['CFO'].shape}")
        print(f"Data shape: {f['data'].shape}")
        print(f"Label shape: {f['label'].shape}")

# 示例调用
if __name__ == '__main__':
    h5_files = [
        './dataset/lora_receive_5_train.h5',
        './dataset/lora_receive_2_train.h5',
        './dataset/lora_receive_3_train.h5',
        './dataset/lora_receive_4_train.h5'
    ]
    output_file = './dataset/4merged_lora_receive_train.h5'
    merge_h5_files(h5_files, output_file)

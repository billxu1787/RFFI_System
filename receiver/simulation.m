filename = 'train.h5'

%% 参数初始化
    num_pkt = 500;
    lora_ind = 1;
    lora_type = 'EoRa';
    sdr_ind = 1;
    rf_freq = 433e6; % 载波频率
    fs = 1e6;           % 采样率
    sf = 7;             % 扩频因子
    bw = 125e3;         % 带宽

    % SDR配置
    config.sdr.type = 'pluto';
    config.sdr.fc = rf_freq; % 中心频率 (Hz)
    config.sdr.FrontEndSampleRate = fs;     % 每秒样本数
    config.FrameLength = 1000000;  % 帧长度 375000
    config.sdr.idx   = 1;
    config.sdr.gain  = 0;

    % LoRa解调配置
    loraphy = LoRaPHY(rf_freq, sf, bw, fs);
    loraphy.has_header = 1;         % 显式头部模式
    loraphy.cr = 3;                 % 码率 = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
    loraphy.crc = 1;                % 启用有效载荷CRC校验和
    loraphy.preamble_len = 8;       % 前导码: 8个基本上调码

    %% 初始化SDR
    [SDR, factor] = sdr_initial(config); % 定义SDR
    preamble = [];
    message = [];
    CFO = [];
    timestamp = [];
    pkt_idx = 1;
    while pkt_idx <= num_pkt
        signal_sdr = SDR();
        data_sdr = transpose(signal_sdr);
        try
            [symbols_d, preamble_phy, CFO_d, ~] = loraphy.demodulate(signal_sdr);
            [message_decoded, checksum] = loraphy.decode(symbols_d);
            num_frames = size(symbols_d, 2);
            timestamp_pkt = datetime('now'); 
            preamble = [preamble preamble_phy];
            message = [message message_decoded];
            CFO = [CFO CFO_d];
            timestamp = [timestamp timestamp_pkt];
            pkt_idx = pkt_idx + num_frames;
            fprintf(['Info: ' num2str(size(preamble, 2)) ' LoRa packets are collected \n'])
        catch
            continue
        end
    end
    release(SDR)

    %% 保存数据
    preamble_iq = [real(preamble); imag(preamble)];

    % 删除现有数据集（如果存在）
    if isfile(filename)
        try
            if h5disp(filename)
                h5delete(filename, '/data');
                h5delete(filename, '/CFO');
                h5delete(filename, '/label');
            end
        catch
            % 如果数据集不存在，则跳过
        end
    end

    % 创建并写入数据集
    h5create(filename, '/data', size(preamble_iq));
    h5write(filename, '/data', preamble_iq);
    h5create(filename, '/CFO', size(CFO));
    h5write(filename, '/CFO', CFO);
    
    % 将 label 转换为字符数组，并转换为 uint8 存储
    label_char = char(label);
    label_size = length(label_char);
    label_data = uint8(label_char); % 转换为 uint8
    h5create(filename, '/label', [label_size 1], 'Datatype', 'uint8');
    h5write(filename, '/label', label_data(:));
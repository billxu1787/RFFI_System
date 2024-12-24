clc
clear all
close all

%% 

num_pkt = 1500;
lora_ind = 1;
lora_type = 'EoRa';
sdr_ind = 1;
rf_freq = 433e6; % carrier frequency
fs = 1e6;           % sampling rate
sf = 7;             % spreading factor
bw = 125e3;         % bandwidth

% SDR config
config.sdr.type = 'pluto';
config.sdr.fc = rf_freq; % Center frequency (Hz)
config.sdr.FrontEndSampleRate = fs;     % Samples per second
config.FrameLength = 1000000;  % Frame length 375000
config.sdr.idx   = 1;
config.sdr.gain  = 0;



% LoRa demodulation config
loraphy = LoRaPHY(rf_freq, sf, bw, fs);
loraphy.has_header = 1;         % explicit header mode
loraphy.cr = 3;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
loraphy.crc = 1;                % enable payload CRC checksum
loraphy.preamble_len = 8;       % preamble: 8 basic upchirps


%% 

[SDR,factor] = sdr_initial(config); % define SDR
label = zeros(1, num_pkt);
preamble = [];
message = [];
CFO = [];
timestamp = [];
pkt_idx = 1;
while pkt_idx <= num_pkt
    
    signal_sdr = SDR();
    data_sdr = transpose(signal_sdr);

    % message_transmit = [1, 3, 5, 7]';
    % snr = 10;
    % signal_sdr = dummy_sdr_sig(message_transmit, snr, loraphy);

    % if  max(real(signal_sdr))>5
    %     fprintf('The received signal power is too high!');
    % end
    
    try
    
        [symbols_d, preamble_phy, CFO_d, ~] = loraphy.demodulate(signal_sdr);
        [message_decoded, checksum] = loraphy.decode(symbols_d); % the last two are
        num_frames = size(symbols_d, 2);
        timestamp_pkt = datetime('now'); 
        preamble = [preamble preamble_phy];
        message = [message message_decoded];
        CFO = [CFO CFO_d];
        timestamp = [timestamp timestamp_pkt];
        label(pkt_idx)=5;
        plot(preamble);
        pkt_idx = pkt_idx + num_frames;
        fprintf(['Info: ' num2str(size(preamble,2)) ' LoRa packets are collected \n'])

    catch
        
        continue

    end
end

release(SDR)

%% Save

preamble_iq = [real(preamble);imag(preamble)];

filename = 'train.h5';

h5create(filename, '/data', size(preamble_iq));
h5write(filename, '/data', preamble_iq);

%h5create(filename, '/message', size(message));
%h5write(filename, '/message', message);

h5create(filename, '/CFO', size(CFO));
h5write(filename, '/CFO', CFO);

h5create(filename, '/label', size(label));  
h5write(filename, '/label', label);


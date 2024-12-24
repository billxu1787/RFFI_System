function sig_channel = dummy_sdr_sig(message_transmit, snr, phy)
    
    symbols = phy.encode(message_transmit);
    sig = phy.modulate(symbols);
    sig_channel = awgn([zeros(100000,1);sig;zeros(100000,1);sig;zeros(100000,1)], snr-10*log10(phy.fs/phy.bw));

end
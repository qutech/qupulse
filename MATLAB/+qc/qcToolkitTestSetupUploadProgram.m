% set upload pulse
% upload_pulse = 'dnp_wait_dbz_4chan';
% upload_pulse = 'decay_j_fid_4chan';
% upload_pulse = 's_pumping_AB_4chan';
% upload_pulse = 'pumping_s_stp';
% upload_pulse = 'pumping_s';
upload_pulse = 'dnp_decay_dbz_fid_4chan';




%% Just an example with the Tabor AWG simulator
% awgctrl('default except offsets')
plsdata.awg.inst.send_cmd(':OUTP:COUP:ALL HV');
args = tunedata.run{1}.(upload_pulse).opts(1).scan.configfn(4).args;
args{end}.window_mapping.A = py.None;
args{end}.window_mapping.B = py.None;
args{end}.window_mapping.DBZFID_A = py.None;
args{end}.window_mapping.DBZFID_B = py.None;
args{end}.operations = {};
feval(args{2:end});

%% reset AWG
awgctrl('default except offsets');

%% check out sequ table wait_4chan
entries = qc.get_sequence_table('wait_4chan', false)
entries{1}{end}
entries{1}{end-1}
entries{1}{end-2}
entries{1}{end-5}


%% check seq table j_fid_4chan
entries = qc.get_sequence_table('j_fid_4chan', false)
entries{1}{end}
entries{1}{end-1}
entries{1}{end-2}
entries{1}{end-5}

%%
entries = qc.get_sequence_table('wait_4chan', true)
disp(entries{1}{1});
disp(entries{2}{1});
% disp(entries{3}{1});
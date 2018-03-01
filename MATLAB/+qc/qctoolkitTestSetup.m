%% Init
savePath = 'Y:\Common\GaAs\Triton 200\Backup\DATA\workspace';

%% Loading
if util.yes_no_input('Really load smdata?', 'n')
	load(fullfile(savePath, 'smdata_recent.mat'));
	fprintf('Loaded smdata\n');
end
load(fullfile(savePath, 'tunedata_recent.mat'));
load(fullfile(savePath, 'plsdata_recent.mat'));
global tunedata
global plsdata

%% Add virtual Alazar channel
% Idea: create a separate qctoolkit instrument
smdata.inst(sminstlookup('ATS9440Python')).channels(9,:) = 'ATSV';
smaddchannel(sminstlookup('ATS9440Python'), 9, smdata.inst(sminstlookup('ATS9440Python')).channels(9, :));
smdata.inst(sminstlookup('ATS9440Python')).data.cache = [];

%%
smloadinst('dummy');
smaddchannel(sminstlookup('dummy'), 1, 'count');

%%
smset('count', 2)
smget('count')

smset('time', now)
smget('time')


%% Setup plsdata from scratch
global plsdata
plsdata = struct( ...
	'path', 'Y:\Cerfontaine\Code\qc-tookit-pulses', ...
	'awg', struct('inst', [], 'hardwareSetup', [], 'sampleRate', 2e9, 'currentProgam', '', 'registeredPrograms', struct(), 'defaultChannelMapping', struct(), 'defaultWindowMapping', struct(), 'defaultParametersAndDicts', {{}}, 'defaultAddMarker', {{}}), ...
  'dict', struct('cache', [], 'path', 'Y:\Cerfontaine\Code\qctoolkit-dicts'), ...
	'qc', struct('figId', 801), ...
	'daq', struct('inst', [], 'defaultOperations', {{}}) ...
	);
plsdata.qc.backend = py.qctoolkit.serialization.FilesystemBackend(plsdata.path);
plsdata.qc.serializer = py.qctoolkit.serialization.Serializer(plsdata.qc.backend);

%% Alazar simulator
smdata.inst(sminstlookup('ATS9440Python')).data.address = 'simulator';
plsdata.daq.inst = py.qctoolkit.hardware.dacs.alazar.AlazarCard(...
	[]...
	);

%% Alazar
smopen('ATS9440Python');
% config = sm_setups.triton_200.AlazarDefaultSettings(); 
% smdata.inst(sminstlookup('ATS9440Python')).data.config = config;
dos('activate lab_master & python -i -c "import atsaverage.client; import atsaverage.gui; card = atsaverage.client.getNetworkCard(''ATS9440'', keyfile_or_key=b''ultra_save_default_key''); window = atsaverage.gui.ThreadedStatusWindow(card); window.start();" &', '-echo')

plsdata.daq.inst = py.qctoolkit.hardware.dacs.alazar.AlazarCard(...
	py.atsaverage.core.getLocalCard(1, 1)...
	);
% alazar = py.qctoolkit.hardware.dacs.alazar.AlazarCard(...
% 	smdata.inst(3).data.py.card ...
% 	);


%% Setup AWG
qc.setup_tabor_awg('realAWG', false, 'simulateAWG', true, 'taborDriverPath', 'Y:\Cerfontaine\Code\tabor');

%% Alazar
qc.setup_alazar_measurements('nQubits', 2, 'nMeasPerQubit', 2, 'disp', true);

%%
% Configure Alazar so the AWG uses the ATS 10MHz reference clock 
py.atsaverage.alazar.ConfigureAuxIO(plsdata.daq.inst.card.handle,...
	                                  py.getattr(py.atsaverage.alazar.AUX_IO_Mode, 'out_pacer'),...
																	  uint64(10));
																	
% Set Base Alazar Config
plsdata.daq.inst.config = py.atsaverage.config.ScanlineConfiguration.parse(sm_setups.triton_200.AlazarDefaultSettings());

%% Load and unload alazar api
% py.atsaverage.alazar.unload
% py.atsaverage.alazar.load('atsapi.dll')

%% AWG default settings
awgctrl('default');

%% Load example pulse (or execute qc-tookit-pulses\matlab\general_charge_scan.m)
charge_scan = qc.load_pulse('charge_scan');

%% Example parameters
parameters = struct('x_start', -1, 'x_stop', 1, 'N_x', 10, 't_meas', 1, ...
                    'W_fast', 1, 'W_slow', 0, ...
                    'X_fast', 1, 'X_slow', 0, ...
                    'Y_fast', 0, 'Y_slow', 1, ...
                    'Z_fast', 0, 'Z_slow', 1, ...
                    'y_start', -1, 'y_stop', 1, 'N_y', 10, 't_wait', 0, 'sample_rate', 2.3, 'meas_time_multiplier', 2, ...
                    'rep_count', 2);

%% Test many of the available Matlab commands
%%
parameters = qc.params_add_delim(parameters, 'charge_scan');

%%
parameters = qc.params_rm_delim(parameters);

%%
test_dict = qc.add_params_to_dict('test', parameters);
qc.save_dict(test_dict);

%%
clearvars test_dict
common_dict = qc.load_dict('test')

%%
instantiated_pulse = qc.instantiate_pulse(charge_scan, 'parameters', parameters);

%%
instantiated_pulse = qc.instantiate_pulse(instantiated_pulse);

%%
qc.plot_pulse(charge_scan, 'parameters', parameters)

%%
qc.plot_pulse(instantiated_pulse)

%%
qc.get_pulse_duration(charge_scan, parameters)

%%
qc.save_pulse(charge_scan, true);

%%
qc.get_pulse_params('charge_scan')

%%
qc.get_pulse_params(charge_scan)

%%
charge_scan = qc.load_pulse('charge_scan');

%%
test_dict = qc.add_params_to_dict('test', struct('x_start', 25, 'x_stop', 16), 'global');
qc.save_dict(test_dict);

%%
parameters2 = struct('x_start', nan, 'x_stop', nan);
parameters2 = qc.params_add_delim(parameters2, 'charge_scan')
qc.join_params_and_dicts(parameters2, 'test')

%%
common_dict = qc.load_dict('common');
qc.save_dict(common_dict);
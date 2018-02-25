%% Setup
global plsdata
plsdata = struct( ...
	'path', 'Y:\Cerfontaine\Code\qc-tookit-pulses', ...
	'awg', struct('inst', [], 'hardwareSetup', [], 'sampleRate', 2e9), ...
  'dict', struct('cache', [], 'path', 'Y:\Cerfontaine\Code\qctoolkit-dicts'), ...
	'qc', struct('figId', 801), ...
	'daq', struct('inst', []) ...
	);
plsdata.qc.backend = py.qctoolkit.serialization.FilesystemBackend(plsdata.path);
plsdata.qc.serializer = py.qctoolkit.serialization.Serializer(plsdata.qc.backend);

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
% The alazar card in THIS example is not controled with qctoolkit and all
% measurement windows defined in the pulses are ignored
function scan = example_scan_no_alazar(hardware_setup, tawg)

global smdata;	
	
%tawg.send_cmd(':INST:SEL 1');
%tawg.send_cmd(':SOUR:MARK:SEL 1; :SOUR:MARK:VOLT:HIGH 1.2');
%tawg.send_cmd(':SOUR:MARK:SEL 1; :SOUR:MARK:STAT OFF');


%set_marker = @() tawg.send_cmd(':ENAB; :TRIG; :SOUR:MARK:SOUR WAVE; :SOUR:MARK:SEL 1; :SOUR:MARK:STAT ON');
%reset_marker = @() tawg.send_cmd(':SOUR:MARK:SEL 1; :SOUR:MARK:STAT OFF');


% use pulse from example files
qctoolkit_location = what('+qctoolkit');
pulse_name = 'table_template';
pulse_location = fullfile(qctoolkit_location.path,...
	'..', '..', 'doc', 'source', 'examples', 'serialized_pulses');


% create struct with parameters
parameters.va = 0;
parameters.vb = 0.5;
parameters.ta = 192;
parameters.tb = 4*19200 - 192;
parameters.tend = parameters.tb + 192;

pulse_length = parameters.tend / tawg.sample_rate(uint64(1));

% we want to play the channel 'A' of the pulse on the channel 'TABOR_A' of
% the hardware_setup. 
channel_mapping.A = 'TABOR_A';

% For a pulse with measurement windows we can rename them here
% window_mapping.meas_in_pulse = 'meas_name'


%% Configure data acquisition with alazar card
sm_alazar_channel = 'ATS1';
sm_alazar_instrument = smchaninst(sm_alazar_channel);
sm_alazar_instrument = sm_alazar_instrument(1);


alazar_config = sm_setups.common.AlazarDefaultSettings();

trigger_range = 1;
trigger_level = 0.01;

alazar_config.trigger_settings.source_1 = 'A';
alazar_config.trigger_settings.level_1 = uint8(128 + 127* (trigger_level / trigger_range));
alazar_config.trigger_settings.slope_1 = 'positive';

switch alazar_config.clock_settings.samplerate
	case 'rate_100MSPS'
		alazar_sample_rate = 100e6;
	otherwise
		error('invalid sample rate (changing the sample rate possibly breaks clock sync)');
end

alazar_downsampling = 1;

masks = {};
masks{1}.type = 'Periodic Mask';
masks{1}.begin = 0;
masks{1}.end = alazar_downsampling; 
masks{1}.period = alazar_downsampling;
masks{1}.channel = 'A';

alazar_config.total_record_size = pulse_length * alazar_sample_rate;
if abs(alazar_config.total_record_size - round(alazar_config.total_record_size)) > 1e-10
	error('total record size is no integer');
end
alazar_config.total_record_size = int64(round(alazar_config.total_record_size));
data_points_per_pulse = alazar_config.total_record_size / alazar_downsampling;

operations = {};
operations{1}.type = 'DS';% downsampling
operations{1}.mask = 1;

alazar_config.masks = masks;
alazar_config.operations = operations;


scan.configfn(1).fn = @smaconfigwrap;
scan.configfn(1).args = {smdata.inst(sm_alazar_instrument).cntrlfn [sm_alazar_instrument 0 99] [] [] alazar_config}; % upload config ?

scan.configfn(2).fn = @smaconfigwrap;
scan.configfn(2).args = {smdata.inst(sm_alazar_instrument).cntrlfn,[sm_alazar_instrument 0 5]}; % write/commit config 

% upload pulse to AWG
scan.configfn(3).fn = @smaconfigwrap;
scan.configfn(3).args = {@qctoolkit.arm_pulse,...
	pulse_name, hardware_setup, parameters, pulse_location...
	'channel_mapping', channel_mapping,...
	'update', true};

scan.loops(1).setchan = [];
scan.loops(1).npoints = data_points_per_pulse;
scan.loops(1).rng = [];
scan.loops(1).ramptime = 0; % = sample rate * mask.period

scan.loops(1).trigfn.fn = @(awg) awg.send_cmd(':TRIG');
scan.loops(1).trigfn.args = {tawg};

scan.loops(2).setchan = [];
scan.loops(2).getchan = {sm_alazar_channel}; % read out buffer
scan.loops(2).npoints = 2;
scan.loops(2).ramptime = [];
scan.loops(2).rng = 1:2;

scan.disp(1).loop = 2;
scan.disp(1).channel = 1;
scan.disp(1).dim = 1;
scan.disp(2).loop = 2;
scan.disp(2).channel = 1;
scan.disp(2).dim = 2;


% arm Alazar before each scan loop(1)
scan.loops(2).prefn(1).fn = @smaconfigwrap;
scan.loops(2).prefn(1).args = {smdata.inst(sm_alazar_instrument).cntrlfn,[sm_alazar_instrument 0 4]}; 

% arm AWG before each scan loop(1) (not really necessary)
scan.loops(2).prefn(2).fn = @smaconfigwrap;
scan.loops(2).prefn(2).args = {@qctoolkit.arm_pulse, pulse_name, hardware_setup};


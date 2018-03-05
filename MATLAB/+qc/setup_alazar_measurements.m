function [mask_prototypes, measurement_map, txt] = setup_alazar_measurements(varargin)	%
	% This function assumes the the first nQubits Alazar channels are hooked
	% up to qubits, the rest are auxiliary channels.
	%
	% Overview over mapping of measurements
	% -----------------------------------------------------------------------
	% measurement name defined in pulse
	% >>> window mapping >>>
	% measurement name defined in hardware setup (1st argument of set_measurement)
	% >>> set_measurement >>>
	% measurement mask (2nd argument of set_measurement, 1st argument of register_mask_for_channel)
	% >>> register_mask_for_channel >>>
	% alazar harware channel (2nd argument of register_mask_for_channel
	% -----------------------------------------------------------------------
	% 
	% Example manual configuration
	% -----------------------------------------------------------------------
	% import py.qctoolkit.hardware.setup.MeasurementMask
	% hws = plsdata.awg.hardwareSetup;
	% daq = plsdata.daq.inst;
	%   any name, give as 2nd arg in window_mapping       alazar mask name
	% hws.set_measurement('A', MeasurementMask(plsdata.daq.inst, 'A'));
	% hws.set_measurement('B', MeasurementMask(plsdata.daq.inst, 'B'));	
	% hws.set_measurement('C', MeasurementMask(plsdata.daq.inst, 'C'));
	% hws.set_measurement('D', MeasurementMask(plsdata.daq.inst, 'D'));	
	% hws.set_measurement('A_B', MeasurementMask(plsdata.daq.inst, 'A'));
	% hws.set_measurement('A_B', MeasurementMask(plsdata.daq.inst, 'B'));
	% 
	%                  alazar mask name, real alazar hardware channel
	% daq.register_mask_for_channel('A', uint64(0));
	% daq.register_mask_for_channel('B', uint64(1));
	% daq.register_mask_for_channel('C', uint64(2));
	% daq.register_mask_for_channel('D', uint64(3));
	% -----------------------------------------------------------------------
	
	global plsdata
	hws = plsdata.awg.hardwareSetup;
	daq = plsdata.daq.inst;
	
	defaultArgs = struct( ...
		'disp', true, ...
		'nMeasPerQubit', 2, ...
		'nQubits', 2 ...
		);
	args = util.parse_varargin(varargin, defaultArgs);
	nAlazarChannels = 4;
	nQubits = args.nQubits;
	nMeasPerQubit = args.nMeasPerQubit;
		
	py.setattr(hws, '_measurement_map', py.dict);
	py.setattr(daq, '_mask_prototypes', py.dict);
	warning('Removing measurement_map and measurement_map might break stuff if previously set. Needs testing.');
	
	for q = 1:nQubits
		for m = 1:nMeasPerQubit
			%                 qubitIndex, measIndex, hwChannel,   auxFlag1
			add_meas_and_mask(q,          m,        q+nQubits-1,  false);
		end
	end
	
	for a = 1:(nAlazarChannels-nQubits)
		for m = 1:nMeasPerQubit
			%                 qubitIndex, measIndex, hwChannel,   auxFlag1
			add_meas_and_mask(a,          m,         a-1,         true);
		end
	end
	
	if args.nQubits > nAlazarChannels
		warning('More than %i qubits not implemented at the moment since Alazar has only %i channels.', nAlazarChannels, nAlazarChannels);
	end
	
	if args.nQubits > 2
		warning('Simultaneous measurements for more than 2 qubits not implemented at the moment.');
	end
	if q == 2
		for m = 1:nMeasPerQubit
			% Q1 Q2           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(1,          m,         2,         false,    2,                3              , false);
			% A1 A2           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(1,          m,         0,         true,     2,                1              , true);
			
			% Q1 A1           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(1,          m,         2,         false,    1,                0              , true);
      % Q1 A2           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(1,          m,         2,         false,    2,                1              , true);
			
			% Q2 A1           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(2,          m,         3,         false,    1,                0              , true);
      % Q2 A2           qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2
			add_meas_and_mask(2,          m,         3,         false,    2,                1              , true);
		end
	end
	
	[mask_prototypes, measurement_map, txt] = qc.get_alazar_measurements('disp', args.disp);	
	
end


function add_meas_and_mask(qubitIndex, measIndex, hwChannel, auxFlag1, secondQubitIndex, secondHwChannel, auxFlag2)
	global plsdata
	
	if nargin < 5
		secondQubitIndex = [];
	end
	
	if nargin < 7
		auxFlag2 = false;
	end
	
	if auxFlag1
		name = 'Aux';
	else
		name = 'Qubit';
	end
	
	if auxFlag2
		name2 = 'Aux';
	else
		name2 = 'Qubit';
	end
	
	if ~isempty(secondQubitIndex)
		measName = sprintf('%s_%i_%s_%i_Meas_%i', name, qubitIndex, name2, secondQubitIndex, measIndex);
		maskName = sprintf('%s_%i_%s_%i_Meas_%i_Mask_%i', name, qubitIndex, name2, secondQubitIndex, measIndex, 1);
		maskName2 = sprintf('%s_%i_%s_%i_Meas_%i_Mask_%i', name, qubitIndex, name2, secondQubitIndex, measIndex, 2);
	else
		measName = sprintf('%s_%i_Meas_%i', name, qubitIndex, measIndex);
		maskName = sprintf('%s_%i_Meas_%i_Mask_%i', name, qubitIndex, measIndex, 1);
	end
	
	plsdata.awg.hardwareSetup.set_measurement(measName, py.qctoolkit.hardware.setup.MeasurementMask(plsdata.daq.inst, maskName));
	plsdata.daq.inst.register_mask_for_channel(maskName, uint64(hwChannel));
	
	if ~isempty(secondQubitIndex)
		plsdata.awg.hardwareSetup.set_measurement(measName, py.qctoolkit.hardware.setup.MeasurementMask(plsdata.daq.inst, maskName2));
		plsdata.daq.inst.register_mask_for_channel(maskName2, uint64(secondHwChannel));
	end
end



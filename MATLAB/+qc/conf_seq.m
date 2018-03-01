function scan = conf_seq(varargin)
	% CONF_SEQ Create special-measure scans with inline qctoolkit pulses
	%
	% Only supports inline scans at the moment (could in principle arm a
	% different program in each loop iteration using prefns but this is not
	% implemented at the moment).
	%
	% This function gets only underscore arguments to be more consistend with
	% qctoolkit. Other variables in this function are camel case.
	%
	% TODO
	% * Implement dbz feedback (use polarizefn to arm another program that has
	%   no defined measurements and thus needs no reconfiguring of the Alazar)
	%
	% --- Outputs -------------------------------------------------------------
	% scan          : special-measure scan
	%
	% --- Inputs --------------------------------------------------------------
	% varargin      : name-value pairs or parameter struct. For a list of
	%                 parameters see the struct defaultArgs below.
	%
	% -------------------------------------------------------------------------
	% (c) 2018/02 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)
	
	global awgdata
	global smdata
	global plsdata
	
	% None of the arguments except pulse_template should contain any python
	% objects to avoid erroneous saving when the scan is executed.
	defaultArgs = struct(...
		... Pulses
		'program_name',         'default_program', ...
		'pulse_template',       'default_pulse', ...
		'parameters_and_dicts', {plsdata.awg.defaultParametersAndDicts}, ...
		'channel_mapping',      plsdata.awg.defaultChannelMapping, ...
		'window_mapping',       plsdata.awg.defaultWindowMapping, ...
		'add_marker',           {plsdata.awg.defaultAddMarker}, ...
		'force_update',         false, ...
		...
		... Measurements
		'operations',           plsdata.daq.defaultOperations, ...
		...
		... Other
		'nrep',                 10, ...        % numer of repetition of pulse
		'dispchans',            'default', ... % list of indices of channels to show
		'saveloop',             0, ...         % save every nth loop
		'dnp',                  false, ...     % enable DNP
		'verbosity',            10 ...         % 0: display nothing, 10: display all
		);
	a = util.parse_varargin(varargin, defaultArgs);
	if ~ischar(a.pulse_template) && ~isstruct(a.pulse_template)
		a.pulse_template = qc.pulse_to_struct(a.pulse_template);
	end
	
	scan = struct();
	nOperations = numel(a.operations);
	
	% Save file and arguments with which scan was created (not stricly necessary)
	scan.data.conf_seq_fn = fileread([mfilename('fullpath') '.m']);
	scan.data.conf_seq_args = a;
	
	% Configure special-measure configfn
	%  * Calling qc.awg_program('add', ...) makes sure the pulse is uploaded
	%    again if any parameters changed.
	%  * If dictionaries were passed as strings, this will automatically
	%    reload the dictionaries and thus use any changes made in the
	%    dictionaries in the meantime.
	%  * The original parameters are saved in scan.data.awg_program. This
	%    includes the pulse_template in json format and all dictionary
	%    entries at the time when the scan was executed.
	%  * If a python pulse_template was passed, this will still save
	%    correctly since it was converted into a Matlab struct above.
	scan.configfn(1).fn = @smaconfigwrap_save_data;
	scan.configfn(1).args = {'awg_program', @qc.awg_program, 'add', a};
	
	% Configure Alazar operations
	% * alazar.update_settings = py.True is automatically set. This results
	%   in reconfiguration of the Alazar which takes a long time. Thus this
	%   should only be done before a scan is started (i.e. in a configfn).
	% * qc.dac_operations('add', a) also resets the virtual channel in
	%   smdata.inst(sminstlookup('ATS9440Python')).data.virtual_channel.
	scan.configfn(2).fn = @smaconfigwrap_save_data;
	scan.configfn(2).args = {'daq_operations', @qc.daq_operations, 'add', a};
	
	% Allow time logging
	% * Update dummy instrument with current time so can get the current time
	%   using a getchan
	scan.loops(1).prefn(1).fn = @smaconfigwrap;
	scan.loops(1).prefn(1).args = {@(chan)(smset('time', now()))};
	
	% Run AWG channel pair 1
	% * Triggers the Alazar
	% * Will later also trigger the RF switches
	% * Will run both channel pairs automatically if they are synced
	%   which they should be by default.
	% * Should be the last prefn so no other channels changed when
	%   measurement starts (really necessary?)
	scan.loops(1).prefn(2).fn = @smaconfigwrap;
	scan.loops(1).prefn(2).args = {@awgctrl, 'run', 1};
	
	% Get AWG information (not needed at the moment)
	% [analogNames, markerNames, channels] = qc.get_awg_channels();
	% [programNames, programs] = qc.get_awg_programs();
	
	% Turns all AWG outpus off after scan
	scan.cleanupfn(1).fn =  @smaconfigwrap;
	scan.cleanupfn(1).args = {@awgctrl, 'off'};
	
	% Configure channels
	scan.loops(1).getchan = cellfun(@(x)('ATSV'), cell(1, numel(a.operations)), 'UniformOutput', false);
	scan.loops(1).getchan(end+1) = {'time'};
	scan.loops(1).setchan = {'count'};
	scan.loops(1).ramptime = [];
	scan.loops(1).npoints = a.nrep;
	scan.loops(1).rng = [];
	
	% Configure display
	scan.disp = [];
	if strcmp(a.dispchans, 'default')
		a.dispchans = 1:min(4, nOperations);
	end
	for l = 1:length(dispchan)
		scan.disp(end+1).loop = 1;
		scan.disp(end).channel = dispchan(l);
		scan.disp(end).dim = 1;
		
		scan.disp(end+1).loop = 1;
		scan.disp(end).channel = dispchan(l);
		scan.disp(end).dim = 2;
	end
	
	if a.saveloop > 0
		scan.saveloop = [1, a.saveloop];
	end
	
	% Add polarization
	if a.dnp
		warning('DNP currently not implemented');
	end
	
	
	
	
	

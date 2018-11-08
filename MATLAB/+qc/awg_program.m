function [program, bool, msg] = awg_program(ctrl, varargin)
	% pulse_template can also be a pulse name. In that case the pulse is
	% automatically loaded.
	
	global plsdata
	hws = plsdata.awg.hardwareSetup;
	daq = plsdata.daq.inst;
	
	program = struct();
	msg = '';
	bool = false;
	
	default_args = struct(...
		'program_name',           'default_program', ...
		'pulse_template',         'default_pulse', ...
		'parameters_and_dicts',   {plsdata.awg.defaultParametersAndDicts}, ...
		'channel_mapping',        plsdata.awg.defaultChannelMapping, ...
		'window_mapping',         plsdata.awg.defaultWindowMapping, ...
		'global_transformation',  plsdata.awg.globalTransformation, ...
		'add_marker',             {plsdata.awg.defaultAddMarker}, ...
		'force_update',           false, ...
		'verbosity',              10 ...
		);
	a = util.parse_varargin(varargin, default_args);
	
	% --- add ---------------------------------------------------------------
	if strcmp(ctrl, 'add')
		[~, bool, msg] = qc.awg_program('fresh', qc.change_field(a, 'verbosity', 0));
		if ~bool || a.force_update
			plsdata.awg.currentProgam = '';
			
			% Deleting old program should not be necessary. In practice however,
			% updating an existing program seemed to crash Matlab sometimes.
			% qc.awg_program('remove', qc.change_field(a, 'verbosity', 10));
			
			a.pulse_template = pulse_to_python(a.pulse_template);
			[a.pulse_template, a.channel_mapping] = add_marker_if_not_empty(a.pulse_template, a.add_marker, a.channel_mapping);
			
			program = qc.program_to_struct(a.program_name, a.pulse_template, a.parameters_and_dicts, a.channel_mapping, a.window_mapping, a.global_transformation);
			plsdata.awg.registeredPrograms.(a.program_name) = program;
			
			if a.verbosity > 9
				fprintf('Program ''%s'' is now being instantiated...', a.program_name);
				tic;
			end
			instantiated_pulse = qc.instantiate_pulse(a.pulse_template, 'parameters', qc.join_params_and_dicts(program.parameters_and_dicts), 'channel_mapping', program.channel_mapping, 'window_mapping', program.window_mapping, 'global_transformation', program.global_transformation);			
			
			if a.verbosity > 9
				fprintf('took %.0fs\n', toc);
				fprintf('Program ''%s'' is now being uploaded...', a.program_name);
				tic
			end
			util.py.call_with_interrupt_check(py.getattr(hws, 'register_program'), program.program_name, instantiated_pulse, pyargs('update', py.True));			
			
			if a.verbosity > 9
				fprintf('took %.0fs\n', toc);
			end
			
			if bool && a.force_update
				msg = ' since update forced';
			else
				msg = '';
			end
			msg = sprintf('Program ''%s'' added%s', a.program_name, msg);
			
			bool = true;
		else
			program = plsdata.awg.registeredPrograms.(a.program_name);
		end
		
	% --- arm ---------------------------------------------------------------
	elseif strcmp(ctrl, 'arm')
		% Call directly before trigger comes, otherwise you might encounter a
		% trigger timeout. Also, call after daq_operations('add')!
		[~, bool, msg] = qc.awg_program('present', qc.change_field(a, 'verbosity', 0));
		if bool			
			% Wait for AWG to stop playing pulse, otherwise this might lead to a
			% trigger timeout since the DAQ is not necessarily configured for the
			% whole pulse time and can return data before the AWG stops playing
			% the pulse.			
			if ~isempty(plsdata.awg.currentProgam)
				waitingTime = min(max(plsdata.awg.registeredPrograms.(plsdata.awg.currentProgam).pulse_duration + plsdata.awg.registeredPrograms.(plsdata.awg.currentProgam).added_to_pulse_duration - (now() - plsdata.awg.triggerStartTime)*24*60*60, 0), plsdata.awg.maxPulseWait);
				if waitingTime == plsdata.awg.maxPulseWait
					warning('Maximum waiting time ''plsdata.awg.maxPulseWait'' = %g s reached.\nIncrease if you experience problems with the data acquistion.', plsdata.awg.maxPulseWait);
				end
				pause(waitingTime);
				% fprintf('Waited for %.3fs for pulse to complete\n', waitingTime);				
			end			
			
			% No longer needed since bug has been fixed
			% qc.workaround_4chan_program_errors(a);		
			
			hws.arm_program(a.program_name);	
	
			plsdata.awg.currentProgam = a.program_name;
			bool = true;
			msg = sprintf('Program ''%s'' armed', a.program_name);
		end
		
  % --- arm ---------------------------------------------------------------
	elseif strcmp(ctrl, 'arm global')
		if ischar(plsdata.awg.armGlobalProgram)
			globalProgram = plsdata.awg.armGlobalProgram;
		elseif iscell(plsdata.awg.armGlobalProgram)
			globalProgram = plsdata.awg.armGlobalProgram{1};
			plsdata.awg.armGlobalProgram = circshift(plsdata.awg.armGlobalProgram, -1);
		else
		  globalProgram = a.program_name;
			warning('Not using global program since plsdata.awg.armGlobalProgram must contain a char or a cell.');
		end		
		
% 		This code outputs the wrong pulses and isn't even faster
% 		registered_programs = util.py.py2mat(py.getattr(hws,'_registered_programs'));
% 		program = registered_programs.(globalProgram);
% 		awgs_to_upload_to = program{4};
% 		dacs_to_arm = program{5};		
% 		for awgToUploadTo = awgs_to_upload_to
% 			awgToUploadTo{1}.arm(globalProgram);
% 		end		
% 		for dacToArm = dacs_to_arm
% 			dacToArm{1}.arm_program(plsdata.awg.currentProgam);
% 		end
		
		qc.awg_program('arm', 'program_name', globalProgram, 'verbosity', a.verbosity, 'arm_global_for_workaround_4chan_program_errors', []);
		
	% --- remove ------------------------------------------------------------
	elseif strcmp(ctrl, 'remove')		
		% Arm the idle program so the program to be remove is not active by
		% any chance (should not be needed - please test more thorougly whether it is needed)
		plsdata.awg.inst.channel_pair_AB.arm(py.None);
		plsdata.awg.inst.channel_pair_CD.arm(py.None);
		
		[~, bool, msg] = qc.awg_program('present', qc.change_field(a, 'verbosity', 0));
		
		if bool
			bool = false;				
			
			if isfield(plsdata.awg.registeredPrograms, a.program_name)
				plsdata.awg.registeredPrograms = rmfield(plsdata.awg.registeredPrograms, a.program_name);
			end
			
			try
				hws.remove_program(a.program_name);
				bool = true;
			catch err
				warning('The following error was encountered when running hardware_setup.remove_program.\nPlease debug AWG commands.\nThis might have to do with removing the current program.\n.Trying to recover by deleting operations.\n%s', err.getReport());
				qc.daq_operations('remove', 'program_name', a.program_name, 'verbosity', 10);
			end
			
			msg = sprintf('Program ''%s'' removed', a.program_name);
		end
		
	% --- clear all ---------------------------------------------------------
	elseif strcmp(ctrl, 'clear all') % might take a long time
		plsdata.awg.registeredPrograms = struct();
		program_names = fieldnames(util.py.py2mat(py.getattr(hws, '_registered_programs')));
		
		bool = true;
		for program_name = program_names.'
			[~, boolNew] = qc.awg_program('remove', 'program_name', program_name{1}, 'verbosity', 10);
			bool = bool & boolNew;
		end
		
		if bool
			msg = 'All programs cleared';
		else
			msg = 'Error when trying to clear all progams';
		end
		
	% --- clear all fast ----------------------------------------------------	
	elseif strcmp(ctrl, 'clear all fast') % fast but need to clear awg manually
		hws.registered_programs.clear();
		py.getattr(daq, '_registered_programs').clear();

	% --- present -----------------------------------------------------------
	elseif strcmp(ctrl, 'present') % returns true if program is present
		bool = py.list(hws.registered_programs.keys()).count(a.program_name) ~= 0;
		if bool
			msg = '';
		else
			msg = 'not ';
		end
		msg = sprintf('Program ''%s'' %spresent', a.program_name, msg);
		
	% --- fresh -------------------------------------------------------------
	elseif strcmp(ctrl, 'fresh') % returns true if program is present and has not changed
		[~, bool, msg] = qc.awg_program('present', qc.change_field(a, 'verbosity', 0));
		
		if isfield(plsdata.awg.registeredPrograms, a.program_name) && bool
			a.pulse_template = pulse_to_python(a.pulse_template);
			[a.pulse_template, a.channel_mapping] = add_marker_if_not_empty(a.pulse_template, a.add_marker, a.channel_mapping);
			
			newProgram = qc.program_to_struct(a.program_name, a.pulse_template, a.parameters_and_dicts, a.channel_mapping, a.window_mapping, a.global_transformation);
			newProgram = qc.get_minimal_program(newProgram);
			% pulse_duration is just a helper field, can recognize whether
			% program has changed without it. Removing it for the equality check
			% below allows for changing the program duration dynamically on the
			% AWG, e.g. for DNP.
			if isfield(newProgram , 'added_to_pulse_duration')
				newProgram  = rmfield(newProgram , 'added_to_pulse_duration');
			end
			
			awgProgram = plsdata.awg.registeredPrograms.(a.program_name);
			awgProgram = qc.get_minimal_program(awgProgram);	
			% pulse_duration is just a helper field, can recognize whether
			% program has changed without it. Removing it for the equality check
			% below allows for changing the program duration dynamically on the
			% AWG, e.g. for DNP.
			if isfield(awgProgram, 'added_to_pulse_duration')
				awgProgram = rmfield(awgProgram, 'added_to_pulse_duration');
			end
			
			bool = isequal(newProgram, awgProgram);
			
			if bool
				msg = '';
			else
				msg = 'not ';
			end
			msg = sprintf('Program ''%s'' is %sup to date (fresh)', a.program_name, msg);
		end
		% 		if ~bool
		% 			util.comparedata(newProgram, awgProgram);
		% 		end
		
	end
	
	if a.verbosity > 9
		fprintf([msg '\n']);
	end
	
	
	
	
function pulse_template = pulse_to_python(pulse_template)
	
	if ischar(pulse_template)
		pulse_template = qc.load_pulse(pulse_template);
	end
	
	if isstruct(pulse_template)
		pulse_template = qc.struct_to_pulse(pulse_template);
	end
	
	
function [pulse_template, channel_mapping] = add_marker_if_not_empty(pulse_template, add_marker, channel_mapping)
	
	if ~iscell(add_marker)
		add_marker = {add_marker};
	end
	
	if ~isempty(add_marker)
		marker_pulse = py.qctoolkit.pulses.PointPT({{0, 1},...
			{py.getattr(pulse_template, 'duration'), 1}}, add_marker);
		pulse_template = py.qctoolkit.pulses.AtomicMultiChannelPT(pulse_template, marker_pulse);
		
		for ii = 1:numel(add_marker)
			channel_mapping.(args.add_marker{ii}) = add_marker{ii};
		end
	end
	
	
	
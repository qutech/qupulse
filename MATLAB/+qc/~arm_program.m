function arm_program(pulse_name_or_template, parameters_and_dicts, varargin)
	% pulse_name_or_template can be a char or a pulse template
	% If it is a pulse template it needs to have pulse name defined
	
	global plsdata
	hws = plsdata.awg.hardwareSetup;
	
	default_args = struct(...
    'channel_mapping', py.None, ...
		'window_mapping',  py.None, ...
		'update',          false, ...
		'verbosity',       10, ...
		'add_marker',      {{}} ...
  );
	
	args = util.parse_varargin(varargin, default_args);
	if ~iscell(args.add_marker)
		args.add_marker = {args.add_marker};
	end
	
	if ischar(pulse_name_or_template)
		program_name = pulse_name_or_template;
	else
		program_name = char(pulse_name_or_template.identifier);
	end
	
	if py.list(hws.registered_programs.keys()).count(program_name) == 0 || args.update
			
		if verbosity > 9
			timerVal = tic;
			if args.update
				txt = 'since update forced';
			else
				txt = 'since program not present';
			end
			
			since parameters changed
			
			fprintf('Updating program %s...instantiating...', program_name, txt);
		end
		
		if nargin < 2
			error('Cannot upload program. Argument ''parameters_and_dicts'' missing.\n');
		end
		
		parse parametersAndDicts

		if ischar(pulse_name_or_template)
			pulse_template = qc.load_pulse(pulse_name_or_template);
		else
			pulse_template = pulse_name_or_template;
		end
		
		% Add marker
		if ~isempty(args.add_marker)			
			marker_pulse = py.qctoolkit.pulses.PointPT({{0, 1},...
				{pulse_template.duration, 1}}, args.add_marker);
			pulse_template = py.qctoolkit.pulses.AtomicMultiChannelPT(pulse_template, marker_pulse);
			
			for ii = 1:numel(args.add_marker)
				args.channel_mapping.(args.add_marker{ii}) = args.add_marker{ii};
			end			
		end
				
		% Plug in parameters
		instantiated_pulse = qc.instantiate_pulse(pulse_template, 'parameters', parameters_and_dicts, 'channel_mapping', args.channel_mapping, 'window_mapping', args.window_mapping);
		
		if verbosity > 9
			timerVal = toc(timerVal);
			fprintf('done (%i s)...uploading...', timerVal);
			tic(timerVal);
		end
		
		% Load program to AWG
		hws.register_program(program_name, instantiated_pulse, pyargs('update', args.update));
		
		if verbosity > 9
			timerVal = toc(timerVal);
			fprintf('done (%i s)\n', timerVal);
			tic(timerVal);
		end
	end
	
	hws.arm_program(program_name);
	
	% Debug
	% alazar = util.py.py2mat(py.getattr(hardware_setup, '_measurement_map')).A{1}.dac
	% prog = util.py.py2mat(py.getattr(alazar, '_registered_programs')).charge_scan
	

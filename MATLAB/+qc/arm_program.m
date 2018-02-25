function arm_program(pulse_name, parameters, varargin)
	
	global plsdata
	hardware_setup = plsdata.awg.hardwareSetup;
	
	default_args = struct(...
    'channel_mapping', py.None, ...
		'window_mapping',  py.None,...
		'update',          false,...
		'add_marker',      {{}} ...
  );
	
	args = util.parse_varargin(varargin, default_args);
	if ~iscell(args.add_marker)
		args.add_marker = {args.add_marker};
	end
	
	if py.list(hardware_setup.registered_programs.keys()).count(pulse_name) == 0 || args.update
		
		if nargin < 2
			error('Cannot upload program. Argument ''parameters'' missing.\n');
		end

		pulse_template = qc.load_pulse(pulse_name);
		
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
		instantiated_pulse = qc.instantiate_pulse(pulse_template, 'parameters', parameters, 'channel_mapping', args.channel_mapping, 'window_mapping', args.window_mapping);
		
		% Load program to AWG
		hardware_setup.register_program(pulse_name, instantiated_pulse, pyargs('update', args.update));
		
	end
	
	hardware_setup.arm_program(pulse_name);
	
	% Debug
	% alazar = util.py.py2mat(py.getattr(hardware_setup, '_measurement_map')).A{1}.dac
	% prog = util.py.py2mat(py.getattr(alazar, '_registered_programs')).charge_scan
	

function arm_pulse(pulse_name, hardware_setup, parameters, pulse_location, varargin)

default_args = struct('channel_mapping', py.None, ...
                      'window_mapping', py.None,...
											'update', false,...
											'add_marker', {{}});

args = util.parse_varargin(varargin, default_args);
if ~iscell(args.add_marker)
	args.add_marker = {args.add_marker};
end

if py.list(hardware_setup.registered_programs.keys()).count(pulse_name) == 0 || args.update

%% LOAD PULSE

backend = py.qctoolkit.serialization.FilesystemBackend(pulse_location);

serializer = py.qctoolkit.serialization.Serializer(backend);

pulse_template = serializer.deserialize(pulse_name);

%% ADD MARKER
if ~isempty(args.add_marker)
	
	marker_pulse = py.qctoolkit.pulses.PointPT({{0, 1},...
		                                         {pulse_template.duration, 1}}, args.add_marker);
	pulse_template = py.qctoolkit.pulses.AtomicMultiChannelPT(pulse_template, marker_pulse);
	
	for ii = 1:numel(args.add_marker)
		args.channel_mapping.(args.add_marker{ii}) = args.add_marker{ii};
	end
	
end


%% INSTANTIATE PULSE (plug in parameters)

sequencer = py.qctoolkit.pulses.Sequencer();

kwargs = pyargs('parameters', parameters,...
       'channel_mapping', args.channel_mapping,...
		   'window_mapping', args.window_mapping);

sequencer.push(pulse_template, kwargs)

instantiated_pulse = sequencer.build();

%% LOAD PROGRAM TO AWG
hardware_setup.register_program(pulse_name, instantiated_pulse, pyargs('update', args.update));

end

hardware_setup.arm_program(pulse_name);


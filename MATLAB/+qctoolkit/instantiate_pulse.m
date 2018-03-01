%% INSTANTIATE PULSE (plug in parameters)
function instantiated_pulse = instantiate_pulse(pulse_template, parameters, varargin)
	
default_args = struct(...
	'channel_mapping', py.None,...
	'window_mapping', py.None);

args = util.parse_varargin(varargin, default_args);

sequencer = py.qctoolkit.pulses.Sequencer();

kwargs = pyargs('parameters', parameters,...
       'channel_mapping', args.channel_mapping,...
		   'window_mapping', args.window_mapping);

sequencer.push(pulse_template, kwargs)

instantiated_pulse = sequencer.build();
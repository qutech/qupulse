function plot_pulse(pulse, varargin)

	global plsdata

	defaultArgs = struct(...
    'sample_rate',     plsdata.awg.sampleRate, ... % in 1/s, converted to 1/ns below
    'parameters',     [], ...
		'channel_mapping', py.None, ...
		'window_mapping' , py.None, ...
    'fig_id',          plsdata.qc.figId, ...
    'clear_fig',       true, ...
    'max_n_points',    1e5 ...
		);

	args = util.parse_varargin(varargin, defaultArgs);

  args.sample_rate = args.sample_rate * 1e-9; % convert to 1/ns
	instantiatedPulse = qc.instantiate_pulse(pulse, 'parameters', args.parameters, 'channel_mapping', args.channel_mapping, 'window_mapping', args.window_mapping);
	
	if ~qc.is_instantiated_pulse(pulse)
		nPoints = qc.get_pulse_duration(pulse, args.parameters) * args.sample_rate * 1e9;
		if nPoints > args.max_n_points
			warning('Number of points %g > %g (maximum number of points). Aborting.\n', ceil(nPoints), args.max_n_points);
			return
		end
	end

	data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiatedPulse, pyargs('sample_rate', args.sample_rate)));
	
	t = data{1}*1e-9;
	
	figure(args.fig_id);
	if args.clear_fig
		clf
	end
	hold on
	
	for chan_name = fieldnames(data{2})'
		plot(t, data{2}.(chan_name{1}), '.-');
	end
	
  xlabel('t(s)');
	legend(fieldnames(data{2})');
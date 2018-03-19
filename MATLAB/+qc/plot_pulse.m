function plot_pulse(pulse, varargin)
	
	global plsdata
	
	defaultArgs = struct(...
		'sample_rate',     plsdata.awg.sampleRate, ... % in 1/s, converted to 1/ns below
		'channel_names',   {{}}, ... % names of channels to plot, all if empty
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
	
	data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiatedPulse, pyargs('sample_rate', args.sample_rate, 'render_measurements', true)));
	
	t = data{1}*1e-9;
	
	channels = data{2};
	measurements = struct();
	for m = data{3}
		if ~isfield(measurements, m{1}{1})
			measurements.(m{1}{1}) = [];
		end
		measurements.(m{1}{1})(end+1, 1:2) = [m{1}{2} m{1}{2}+m{1}{3}] * 1e-9;
	end
	
	figure(args.fig_id);
	if args.clear_fig
		clf
	end
	hold on
	
	legendEntries = {};
	legendHandles = [];
	
	for meas_name = fieldnames(measurements)'
		h = plot(measurements.(meas_name{1}).', measurements.(meas_name{1}).'*0, 'lineWidth', 4, 'displayName', ['Meas: ' meas_name{1}]);
		legendHandles(end+1) = h(1);
		legendEntries{end+1} = ['Meas: ' meas_name{1}];
	end
	
	for chan_name = fieldnames(channels)'
		if isempty(args.channel_names) || any(cellfun(@(x)(strcmp(x, chan_name{1})), args.channel_names))
			h = util.rectplot(t, channels.(chan_name{1}), '.-');
			legendHandles(end+1) = h(1);
			legendEntries{end+1} = ['Chan: ' chan_name{1}];
		end
	end		
	
	xlabel('t(s)');
	legend(legendHandles, legendEntries);
	
end
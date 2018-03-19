function plot_pulse(pulse, varargin)
	
	global plsdata
	
	defaultArgs = struct(...
		'sample_rate',       plsdata.awg.sampleRate, ... % in 1/s, converted to 1/ns below
		'channel_names',     {{}}, ... % names of channels to plot, all if empty
		'parameters',        [], ...
		'channel_mapping',   py.None, ...
		'window_mapping' ,   py.None, ...
		'fig_id',            plsdata.qc.figId, ...
		'clear_fig',         true, ...
		'charge_diagram',    {{'X', 'Y'}}, ...
		'lead_points',       1e-3*[-4 -1; -1 -2; 0 -4; 4 0; 2 1; 1 4], ...
		'special_points',    struct('M', [0 0], 'R1', [-2.5e-3 -3.75e-3], 'R2', [-2e-3 1e-3], 'S', [-2e-3 -1e-3], 'Tp', [1.75e-3 0], 'STp', [1e-3 -1e-3]), ...
		'plotRange',         [-8e-3 8e-3], ...
		'max_n_points',      1e5 ...
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
	
	try
		data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiatedPulse, pyargs('sample_rate', args.sample_rate, 'render_measurements', true)));
	catch err
		warning('The following error occurred when plotting. This might have to do with Python dicts not being convertable to Matlab because of illegal struct field name:\n%s', err.getReport())
	end
	
	t = data{1}*1e-9;
	
	channels = data{2};
	if ~isempty(args.plotRange)
		for chan_name = fieldnames(channels)'
			channels.(chan_name{1}) = util.clamp(channels.(chan_name{1}), args.plotRange);
		end
	end
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
	if ~isempty(args.charge_diagram)
		subplot(121);
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
	
	if ~isempty(args.plotRange)
		title(['Plot range: ' sprintf('%g ', args.plotRange)]);
	end
	xlabel('t(s)');
	legend(legendHandles, legendEntries);	
	
	
	if ~isempty(args.charge_diagram)
		subplot(122);
		hold on
		userData = get(gca, 'UserData');
		
		if isempty(userData) || ~isstruct(userData) || ~isfield(userData, 'leadsPlotted') || ~userData.leadsPlotted			
			color = [1 1 1]*0.7;
			lineWidth = 3;
			
			plot(args.lead_points(1:3,1), args.lead_points(1:3,2), '-', 'lineWidth', lineWidth, 'color', color);
			plot(args.lead_points(4:6,1), args.lead_points(4:6,2), '-', 'lineWidth', lineWidth, 'color', color);
			plot(args.lead_points([2 5],1), args.lead_points([2 5],2), '--', 'lineWidth', lineWidth, 'color', color);
			
			offset = abs(max(args.lead_points(:))-min(args.lead_points(:)))*0.05;
			
			for name = fieldnames(args.special_points)'
				xy = args.special_points.(name{1});
				plot(xy(1), xy(2), 'g.', 'markerSize', 24, 'color', color);
				text(xy(1), xy(2)+offset, name{1}, 'horizontalAlignment', 'center', 'color', color);
			end
			
			set(gca, 'UserData', struct('leadsPlotted', true));
		end
		
		 plot(channels.(args.charge_diagram{1}), channels.(args.charge_diagram{2}), '.-', 'MarkerSize', 16);
				
	end
	
end
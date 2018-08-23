function check_pulse_4chan_parameter_dependency(pulse, check_parameter, check_values, varargin)

defaultArgs = struct(...
		'parameters',          struct() ...
);
	
args = util.parse_varargin(varargin, defaultArgs);

for k = 1:numel(check_values)
  args.parameters.(check_parameter) = check_values(k);
  args.fig_id = 2303+k;
  args.subplots = [121 122] + [k k];
  qc.plot_pulse_4chan(pulse, args);
end

end
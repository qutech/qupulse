function check_pulse_4chan_parameter_dependency(pulse, check_parameter, check_values, varargin)

defaultArgs = struct(...
  'sample_rate',         plsdata.awg.sampleRate, ... % in 1/s, converted to 1/ns below
  'channel_mapping',     py.None, ...
  'window_mapping' ,     py.None, ...
  'parameters',          struct() ...
  );

args = util.parse_varargin(varargin, defaultArgs);
args.sample_rate = args.sample_rate * 1e-9; % convert to 1/ns

figure();

for k = 1:numel(check_values)
  args.parameters.(check_parameter) = check_values(k);
  
  instantiatedPulse = qc.instantiate_pulse(pulse, 'parameters', args.parameters, 'channel_mapping', args.channel_mapping, 'window_mapping', args.window_mapping);
  data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiatedPulse, pyargs('sample_rate', args.sample_rate, 'render_measurements', true)));
  
  subplot(1, numel(check_values), k);
  plot(data)
  
end

end
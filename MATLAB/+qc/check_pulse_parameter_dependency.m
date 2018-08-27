function check_pulse_parameter_dependency(pulse, check_parameter, check_values, varargin)
% plot function to visualize the influence of one pulse parameter on the
% pulse shape:
% ----------------------------------------------------------------------
% input:
% pulse               : loaded qctoolkit pulse
% check_paramter      : name of the paramter under investigation
% check_values        : values the parameter is set to, array []
% varargin            : standard varargin that one also uses for
%                       qc.plot_pulse
% ----------------------------------------------------------------------
% written by Marcel Meyer 08|2018 (marcel.meyer1@rwth-aachen.de)

global plsdata

defaultArgs = struct(...
  'sample_rate',         plsdata.awg.sampleRate, ... % in 1/s, converted to 1/ns below
  'channel_mapping',     py.None, ...
  'window_mapping' ,     py.None, ...
  'parameters',          struct(), ...
  'removeTrigChans',     true, ...
  'figID',               2018 ...
  );

args = util.parse_varargin(varargin, defaultArgs);

if isempty(args.channel_mapping) || args.channel_mapping == py.None

  args.channel_mapping = py.dict(py.zip(pulse.defined_channels, pulse.defined_channels));
end

args.sample_rate = args.sample_rate * 1e-9; % convert to 1/ns

figure(args.figID);
clf;

for k = 1:numel(check_values)
  args.parameters.(check_parameter) = check_values(k);
  
  instantiatedPulse = qc.instantiate_pulse(pulse, 'parameters', args.parameters, 'channel_mapping', args.channel_mapping, 'window_mapping', args.window_mapping);
  data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiatedPulse, pyargs('sample_rate', args.sample_rate, 'render_measurements', true)));
  
  subplot(1, numel(check_values), k);
  
  hold on;
  
  if args.removeTrigChans
    data{2} = rmfield(data{2}, 'MTrig');
    data{2} = rmfield(data{2}, 'M1');
    data{2} = rmfield(data{2}, 'M2');
  end
  
  channelNames = fieldnames(data{2});
  
  for channelInd = 1:numel(channelNames)
    plot(data{1}*1e-9, data{2}.(channelNames{channelInd}));
  end
  hold off;
  legend(channelNames);
  xlabel('t(s)');
  title(sprintf('%s = %.2d', check_parameter, check_values(k)), 'interpreter', 'none');
end

end
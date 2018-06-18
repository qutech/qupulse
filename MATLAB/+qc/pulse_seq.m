function [pulse, args] = pulse_seq(pulses, varargin)
	
% PULSE_SEQ Summary
%   Dynamically sequence qctoolkit pulses
%
% --- Outputs -------------------------------------------------------------
% pulse         : Sequenced pulse template
% args          : Struct of all input arguments, including pulses
%
% --- Inputs --------------------------------------------------------------
% pulses        : Cell of pulse identifiers or pulse templates in sequence order
% varargin      : Name-value pairs or struct
%
% -------------------------------------------------------------------------
% (c) 2018/06 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)
global plsdata

defaultArgs = struct( ...
	'repetitions',                  ones(1, numel(pulses)), ... % Repetition for each pulse
	'fill_param',                  '',		  								... % Not empty: Automatically add fill_pulse to achieve total time given by this parameter.
	                                                        ... % 'auto': Determine fill time automatically for efficient upload on Tabor AWG
	'fill_pulse_param',            'wait___t',              ... % Name of pulse parameter to use for total fill time
	'fill_pulse',									 'wait',                  ... % Pulse template or identifier of pulse to use for filling (added to beginning of pulse sequence)
	'measurements',                [],							   			... % Empty:     Do not define any additional readout. 
                                                          ... % Otherwise: Argument #1 to pyargs('measurements', #1) of SequencePT without fill
	'prefix',                      '' ,											... % Prefix to add to each pulse parameters
	'identifier',                  ''	, 										... % Empty:     Do not add an identifier
                                                          ... % Otherwise: Name of the final pulse
  'sampleRate',          uint64(plsdata.awg.sampleRate),  ... % In SI units (Sa/s), should be uint
	'minSamples',                 uint64(192),              ... % Minimum number of samples for fill pulse, should be uint
  'sampleQuantum',              uint64(16)                ... % Sample increments for fill pulse, should be uint
	);
args = util.parse_varargin(varargin, defaultArgs);
args.pulses = pulses;
args.sampleRate = args.sampleRate/1e9; % in GSa/s

% Specific 


% Load and repeat pulses
for k = 1:numel(pulses)  
	if ischar(pulses{k})
		pulses{k} = qc.load_pulse(pulses{k});
	end
	
  if args.repetitions(k) > 1
    pulses{k} = py.qctoolkit.pulses.RepetitionPT(pulses{k}, args.repetitions(k));
	end  
end

% Sequence pulses
if ~isempty(args.measurements)
	pulse = py.qctoolkit.pulses.SequencePT(pulses{:}, pyargs('measurements', args.measurements));
else
	pulse = py.qctoolkit.pulses.SequencePT(pulses{:});
end

% Add fill if fill_param not empty
if ~isempty(args.fill_param)
	duration = pulse.duration;
	if qc.is_instantiated_pulse(args.fill_pulse)		
		fill_pulse = args.fill_pulse;
	else
		fill_pulse = qc.load_pulse(args.fill_pulse);
	end
	if strcmp(args.fill_param, 'auto')
		minDuration = args.minSamples/args.sampleRate;
		durationQuantum = args.sampleQuantum/args.sampleRate;

		fill_time =																																												...
			py.sympy.ceiling(																																								... If duration > minDuration, get the next higher 
			  ( py.sympy.Max(duration.sympified_expression-minDuration, durationQuantum) )/durationQuantum  ... integer number of durationQuantum above minDuration
			)*durationQuantum																																								... but with minimum of one durationQuantum so pulse length not 0
			+ minDuration																																										... Enforce that duration always longer than minDuration
			- duration.sympified_expression;
	
	else
		fill_time = args.fill_param - duration;
	end
	
	fill_pulse = py.qctoolkit.pulses.MappingPT( ...
			pyargs( ...
				'template',  fill_pulse, ...			
				'parameter_mapping', qc.dict(args.fill_pulse_param, fill_time), ...
				'allow_partial_parameter_mapping', true ...
				) ...
			);
	pulse = py.qctoolkit.pulses.SequencePT(fill_pulse, pulse);
end

% Add prefix to all pulse parameters (if not empty)
if ~isempty(args.prefix)
	parameters = qc.get_pulse_params(pulse);
	parameter_mapping = cell(1, 2*numel(parameters));
	for k = 1:numel(parameters)
		parameter_mapping{1, 2*k-1} = parameters{k};
		parameter_mapping{1, 2*k} = strcat(args.prefix, parameters{k});
	end

	pulse = py.qctoolkit.pulses.MappingPT( ...
		pyargs( ...
			'template',  pulse, ...			
			'parameter_mapping', qc.dict(parameter_mapping{:}), ...
			'allow_partial_parameter_mapping', true ...
			) ...
		);
end

% Add pulse identifier of identifier not empty
if ~isempty(args.identifier)	
	pulse = py.qctoolkit.pulses.SequencePT( ...
		pulse, ...
		pyargs('identifier', args.identifier) ...
		);
end


function [optimalTime, addTime, optimalNsamp, actualTime] = get_optimal_pulse_time(pulse, varargin)
		
	if ischar(pulse)
		pulse = qc.load_pulse(pulse);
	end
	
	actualTime = qc.get_pulse_duration(pulse, qc.join_params_and_dicts(varargin{:}));
	[optimalTime, addTime, optimalNsamp] = qc.get_optimal_awg_time(actualTime);
	
	
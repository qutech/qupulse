function [program, channels] = get_program(pulse, varargin)
	
	if ischar(pulse)
		pulse = qc.load_pulse(pulse);
	end
	
	instantiated_pulse = qc.instantiate_pulse(pulse, 'parameters', qc.join_params_and_dicts(varargin{:}));
		
	tmp = py.qctoolkit.hardware.program.MultiChannelProgram(instantiated_pulse);
	program = py.next(py.iter(tmp.programs.values()));
	channels = py.next(py.iter(tmp.programs.keys()));
	
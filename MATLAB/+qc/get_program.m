function [program, channels] = get_program(pulse, varargin)
	
	instantiated_pulse = qc.instantiate_pulse(pulse, varargin{:});
		
	tmp = py.qctoolkit.hardware.program.MultiChannelProgram(instantiated_pulse);
	program = py.next(py.iter(tmp.programs.values()));
	channels = py.next(py.iter(tmp.programs.keys()));
	
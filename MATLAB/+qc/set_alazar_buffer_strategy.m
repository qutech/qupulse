function set_alazar_buffer_strategy(strategy, varargin)
	% SET_ALAZAR_BUFFER_STRATEGY  sets the strategy to determine the buffer
	% size the alazar card uses.
	%   strategy =
	%   {'force_buffer_size'|'avoid_single_buffer'|'one_buffer_per_window'|'none'}
	%  
	%  strategy = 'none'
	%    Uses default from python
	%
	%  strategy = 'force_buffer_size'
	%    Requires the argument 'target_size'
	%
	%  strategy = 'avoid_single_buffer'
	%    Optional argument 'target_size'
	%  
	%  strategy = 'one_buffer_per_window'
	%    Uses the gcd of all measurement window periods. Usefull for charge
	%    scans with lots of averaging
	
	import py.qupulse.hardware.dacs.alazar.AvoidSingleBufferAcquisition
	import py.qupulse.hardware.dacs.alazar.OneBufferPerWindow
	import py.qupulse.hardware.dacs.alazar.ForceBufferSize
	
	
	global plsdata
	
	switch lower(strategy)
		case 'none'
			py_strategy = py.None;
		
		case 'force_buffer_size'
			args = util.parse_varargin(varargin{:});
			if ~isfield(args, 'target_size')
				error('qc:set_alazar_buffer_strategy:missing','The buffer strategy "force_buffer_size" requires "target_size".')
			end
			py_strategy = ForceBufferSize(struct2pyargs(args));
			
			
		case 'avoid_single_buffer'
			default_args = struct('target_size', py.int(2^22));
			args = util.parse_varargin(default_args, varargin{:});
			py_strategy = AvoidSingleBufferAcquisition(ForceBufferSize(struct2pyargs(args)));
			
		case 'one_buffer_per_window'
			py_strategy = py.qupulse.hardware.dacs.alazar.OneBufferPerWindow();
			
		otherwise
			error('qc:set_alazar_buffer_strategy:unknown', 'Unknown buffer strategy "%s"', strategy);
	end
	
	plsdata.daq.inst.buffer_strategy = py_strategy;
	fprintf('Set buffer strategy to %s.\n', char(py.repr(py_strategy)));
end

function py_args = struct2pyargs(s)
	c = [fieldnames(s), struct2cell(s)]';
	py_args = pyargs(c{:});
end
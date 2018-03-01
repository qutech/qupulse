function [operations, bool, msg] = daq_operations(ctrl, varargin)
	
	global plsdata smdata
	hws = plsdata.awg.hardwareSetup;
	daq = plsdata.daq.inst;
	instIndex = sminstlookup('ATS9440Python');
	
	program = struct();
	msg = '';
	bool = false;
	
	default_args = struct(...
		'program_name',         'default_program', ...
		'operations',           {plsdata.daq.defaultOperations}, ...
		'verbosity',            10 ...
		);
	a = util.parse_varargin(varargin, default_args);
	operations = a.operations;
	
	% --- add ---------------------------------------------------------------
	if strcmp(ctrl, 'add')		
		
		smdata.inst(instIndex).data.virtual_channel = struct( ...
			'data', struct(), ...
			'extractNextScanline', true, ...
			'operationsInd', 1, ...
			'operations', {a.operations} ...
			);
		
		% alazar.update_settings = py.True is automatically set if
		% register_operations is executed. This results in reconfiguration
		% of the Alazar which takes a long time.		
		daq.register_operations(a.program_name, qc.operations_to_python(a.operations));
		msg = sprintf('Operations for program ''%s'' added', a.program_name);
		bool = true;
	
	% --- remove ------------------------------------------------------------
	elseif strcmp(ctrl, 'remove')
		try
			smdata.inst(instIndex).data.virtual_channel = struct( ...
				'data', struct(), ...
				'extractNextScanline', false, ...
				'operationsInd', [], ...
				'operations', {{}} ...
				);
			daq.delete_program(a.program_name);
			msg = sprintf('Operations for program ''%s'' deleted', a.program_name);
			bool = true;
		catch err
			if util.str_contains(err.message, 'KeyError')
				msg = sprintf('Cannot delete program ''%s'' from Alazar configuration since it was not present in the Alazar configuration.\n', a.program_name);
				bool = true;
			else
				msg = '';
				warning(err.getReport());
				bool = false;
			end			
		end		
		
	end
	
	if a.verbosity > 9
		fprintf([msg '\n']);
	end
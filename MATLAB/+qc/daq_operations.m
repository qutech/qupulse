function [output, bool, msg] = daq_operations(ctrl, varargin)
	
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
	output = a.operations;
	
	% --- add ---------------------------------------------------------------
	if strcmp(ctrl, 'add') % output is operations		 		
		% qc.daq_operations('remove', qc.change_field(a, 'verbosity', 0));

		smdata.inst(instIndex).data.virtual_channel = struct( ...
			'operations', {a.operations} ...
			);
		
		% alazar.update_settings = py.True is automatically set if
		% register_operations is executed. This results in reconfiguration
		% of the Alazar which takes a long time.		
		daq.register_operations(a.program_name, qc.operations_to_python(a.operations));
		msg = sprintf('Operations for program ''%s'' added', a.program_name);
		bool = true;
	
  % --- set length --------------------------------------------------------
	elseif strcmp(ctrl, 'set length') % output is length
		% Operations need to have been added beforehand
		output = qc.daq_operations('get length', a);		
		smdata.inst(instIndex).cntrlfn([instIndex nan 999], output);
		
  % --- get length --------------------------------------------------------
	elseif strcmp(ctrl, 'get length') % output is length
		% Operations need to have been added beforehand
		masks = util.py.py2mat(py.getattr(daq, '_registered_programs'));
		masks = util.py.py2mat(masks.(a.program_name));
		masks = util.py.py2mat(masks.masks);
		output = [];
		for k = 1:numel(masks)
			output(k) = util.py.py2mat(size(masks{k}.length));
		end	
		if isempty(output)
			warning('No masks configured');
		end
		
	% --- get ---------------------------------------------------------------
	elseif strcmp(ctrl, 'get programs') % output is registered programs
		% Operations need to have been added beforehand
		% masks = util.py.py2mat(daq.config.masks); % this worked sometimes but sometimes not
		output = util.py.py2mat(py.getattr(daq, '_registered_programs'));
	
	% --- remove ------------------------------------------------------------
	elseif strcmp(ctrl, 'remove') % output is operations		
		smdata.inst(instIndex).data.virtual_channel = struct( ...
			'operations', {{}} ...
			);
		programs = fieldnames(qc.daq_operations('get programs'));
		if any(cellfun(@(x)(strcmp(x, a.program_name)), programs))
			daq.delete_program(a.program_name);
			msg = sprintf('Operations for program ''%s'' deleted', a.program_name);
			bool = true;
		else
			msg = sprintf('Operations for program ''%s'' were not registered', a.program_name);
			bool = true;
		end		
		
  % --- clear all ---------------------------------------------------------
  elseif strcmp(ctrl, 'clear all')
		alazarPackage = py.importlib.import_module('qctoolkit.hardware.dacs.alazar');
		py.setattr(daq, '_registered_programs', py.collections.defaultdict(alazarPackage.AlazarProgram));
		bool = true;
		msg = 'All programs cleared from DAQ';
		
	end
	
	if a.verbosity > 9
		fprintf([msg '\n']);
	end
function [output, bool, msg] = daq_operations(ctrl, varargin)
	
	global plsdata smdata
	hws = plsdata.awg.hardwareSetup;
	daq = plsdata.daq.inst;
	instIndex = sminstlookup(plsdata.daq.instSmName);
	
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
		% Call before qc.awg_program('arm')!

		smdata.inst(instIndex).data.virtual_channel = struct( ...
			'operations', {a.operations} ...
			);
		
		% alazar.update_settings = py.True is automatically set if
		% register_operations is executed. This results in reconfiguration
		% of the Alazar which takes a long time. Thus, we avoid registering
		% operations if the last armed program is the same as the currently
		% armed program. We know plsdata.awg.currentProgam contains the last
		% armed program since qc.daq_operations should be called before 
		% qc.awg_program('arm'). 
		if plsdata.daq.reuseOperations && ~plsdata.daq.operationsExternallyModified && strcmp(plsdata.awg.currentProgam, a.program_name)
			msg = sprintf('Operations from last armed program ''%s'' reused.\n  If an error occurs, try executing another program\n  first to update the operations.', plsdata.awg.currentProgam);
		else
			daq.register_operations(a.program_name, qc.operations_to_python(a.operations));
			msg = sprintf('Operations for program ''%s'' added', a.program_name);			
						
			if plsdata.daq.operationsExternallyModified
				plsdata.daq.inst.update_settings = py.True;
			end
			
			plsdata.daq.operationsExternallyModified = false;
			% qc.workaround_alazar_single_buffer_acquisition();
		end
		bool = true;
	
  % --- set length --------------------------------------------------------
	elseif strcmp(ctrl, 'set length') % output is length
		% Operations need to have been added beforehand
		output = qc.daq_operations('get length', a);		
		smdata.inst(instIndex).cntrlfn([instIndex nan 999], output);
		
  % --- get length --------------------------------------------------------
	elseif strcmp(ctrl, 'get length') % output is length
		% Operations need to have been added beforehand
        mask_maker = py.getattr(daq, '_make_mask');
		masks = util.py.py2mat(py.getattr(daq, '_registered_programs'));
		masks = util.py.py2mat(masks.(a.program_name));
		operations = masks.operations;		
		masks = util.py.py2mat(masks.masks(mask_maker));
        
		
		maskIdsFromOperations = cellfun(@(x)(char(x.maskID)), util.py.py2mat(operations), 'UniformOutput', false);
		maskIdsFromMasks = cellfun(@(x)(char(x.identifier)), util.py.py2mat(masks), 'UniformOutput', false);
		
		output = [];
		for k = 1:length(operations)
			maskIndex = find(  cellfun(@(x)(strcmp(x, maskIdsFromOperations{k})), maskIdsFromMasks) );
			if numel(maskIndex) ~= 1
				error('Found several masks with same identifier. Might be a problem in qctoolkit or in this function.');
			end
			
			if isa(operations{k}, 'py.atsaverage._atsaverage_release.ComputeDownsampleDefinition')
				output(k) = util.py.py2mat(size(masks{maskIndex}.length));
			elseif isa(operations{k}, 'py.atsaverage._atsaverage_release.ComputeRepAverageDefinition')
			  n = util.py.py2mat(masks{maskIndex}.length.to_ndarray);
				if any(n ~= n(1))
					error('daq_operations assumes that all masks should have the same length if using ComputeRepAverageDefinition.');
				end				
				output(k) = n(1);
			else
				error('Operation ''%s'' not yet implemented', class(operations{k}));
			end
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
		% Should not call this usually. Call qc.awg_program('remove') instead.
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
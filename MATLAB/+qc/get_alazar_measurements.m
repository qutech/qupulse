function [mask_prototypes, measurement_map, txt] =  get_alazar_measurements(varargin)
	
	global plsdata
	hws = plsdata.awg.hardwareSetup;
	daq = plsdata.daq.inst;
	
	defaultArgs = struct( ...
		'disp', true ...
		);
	args = util.parse_varargin(varargin, defaultArgs);
	
	mask_prototypes = util.py.py2mat(daq.mask_prototypes);
	measurement_map = util.py.py2mat(py.getattr(hws,'_measurement_map'));
	
	txt = sprintf('%-25s  %-25s  %-25s\\n', 'Measurement', 'Mask', 'Hardware Channel');
	txt = strcat(txt, [ones(1,70)*'-' '\n']);
	
	for measName = fieldnames(measurement_map)'
		masks = measurement_map.(measName{1});
		for k = 1:numel(masks)
			maskName = char(masks{k}.mask_name);
			txt = strcat(txt, sprintf('%-25s  %-25s  %-25i\\n', measName{1}, maskName, mask_prototypes.(maskName){1}));
		end		
	end
	
	txt = strcat(txt, [ones(1,70)*'-' '\n']);
	
	if args.disp
		fprintf(txt);
	end
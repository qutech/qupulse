function [programNames, programs] = get_awg_programs()
	
	global plsdata
	
	programs = util.py.py2mat(plsdata.awg.hardwareSetup.registered_programs);	
	programNames = fieldnames(programs);	
	
	if ~isempty(setdiff(fieldnames(rmfield(plsdata.awg.registeredPrograms, 'currentProgam')), programNames))
		warning('''plsdata.awg.registeredPrograms'' out of sync with ''plsdata.awg.hardwareSetup.registered_programs''. Clear all programs by executing qc.awg_program(''clear all'') to remedy.');
	end
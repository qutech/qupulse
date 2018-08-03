function workaround_4chan_program_errors(a)
	% For some 4 channel programs, running another 2 channel program which
	% has marker channels on both AWGs beforehand, leads to erroneous voltage
	% outputs, even though the (advanced) sequence tables stored by qctoolkit
	% do not change. This is true even if qctoolkit is forced to reupload
	% the sequence tables of the 4 channel program.
	%
	% To repdroduce this error reset the AWG, then run
	% EITHER
	%  1) tune('resp'): correct output
	%  2) tune('lead', 2): correct output
	%  3) tune('resp'): erroneous output - if omitted, 5) gives erroneous output
	%  4) tune('line', 1): correct output
	%  5) tune('resp'): correct output
	% OR
	%  1) tune('lead', 2): correct output
	%  2) tune('comp'): correct output
	%  3) tune('resp'): erroneous output
	%
	% I (Pascal) found out this can be circumvented by arming the erroneous
	% program and then arming the idle program manually. Next, the erroneous
	% program can be run and now yields the correct result.
	%
	% This bug has been fixed now by adding the following lines in tabor.py
	%		self.device.send_cmd('SEQ:DEL:ALL')
	%   self._sequencer_tables = []
	%   self.device.send_cmd('ASEQ:DEL')
	%   self._advanced_sequence_table = []
	
	warning('No longer needed since bug has been fixed');
	
	% 	global plsdata
	% 	
	% 	if ~strcmp(plsdata.awg.currentProgam, a.program_name) && (~isfield(a, 'arm_global_for_workaround_4chan_program_errors'))
	% 		tic
	% 		
	% 		hws = plsdata.awg.hardwareSetup;
	% 		known_awgs = util.py.py2mat(hws.known_awgs);
	% 		
	% 		for k = 1:numel(known_awgs)
	% 			if any(cellfun(@(x)(strcmp(x, a.program_name)), fieldnames(util.py.py2mat(py.getattr(known_awgs{k}, '_known_programs')))))
	% 				known_awgs{k}.change_armed_program(a.program_name);
	% 			end
	% 		end
	% 		
	% 		for k = 1:numel(known_awgs)
	% 			known_awgs{k}.change_armed_program(py.None);
	% 		end
	% 		
	% 		fprintf('qc.workaround_4chan_program_errors executed...took %.0fs\n', toc);
	% 	end
function disp_awg_seq_table(varargin)
	
	global plsdata
	
	defaultArgs = struct(...
		'programName', plsdata.awg.currentProgam, ...
		'advancedSeqTableFlag', false ...
		);
	args = util.parse_varargin(varargin, defaultArgs);
	
	
	
	seq_table = qc.get_sequence_table(args.programName, args.advancedSeqTableFlag);
	
	disp('   ');
	disp('[i] Table 1 is for channel pair AB and table 2 for channel pair CD.');
	disp('   ');
	
	for k = 1:2
		if ~args.advancedSeqTableFlag
			
			disp(sprintf('--- table %d -----------------', k));
			for n = 1:length(seq_table{k})
				disp(sprintf(' -- sub table %d ---------', n));
				
				counter_disp_flag = false;
				tmpEntry = seq_table{k}{n}{1};
				disp(tmpEntry);
				counter = 0;
				for i=1:length(seq_table{k}{n})
					if isequal(tmpEntry, seq_table{k}{n}{i})
						counter = counter+1;
					else
						disp(sprintf('    rep = %d', counter));
						tmpEntry = seq_table{k}{n}{i};
						disp(tmpEntry);
						counter = 0;
						counter_disp_flag = true;
					end
				end
				if counter_disp_flag == false
					disp(sprintf('    rep = %d', counter));
				end
				disp('-----------------------------')
			end
			
			
			
		else
			
			disp(sprintf('--- table %d -----------------', k));
			
			counter_disp_flag = false;
			tmpEntry = seq_table{k}{1};
			disp(tmpEntry);
			counter = 0;
			for i=1:length(seq_table{k})
				if isequal(tmpEntry, seq_table{k}{i})
					counter = counter+1;
				else
					disp(sprintf('    rep = %d', counter));
					tmpEntry = seq_table{k}{i};
					disp(tmpEntry);
					counter = 0;
					counter_disp_flag = true;
				end
			end
			if counter_disp_flag == false
				disp(sprintf('    rep = %d', counter));
			end
			disp('-----------------------------')
		end
		
		
	end
	
end
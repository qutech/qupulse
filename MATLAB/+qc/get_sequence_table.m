function seq_table = get_sequence_table(program_name, advanced_seq_table_flag, awg_channel_pair_identifiers, verbosity, return_python_list)
% GET_SEQUENCE_TABLE Get sequence table of program on Tabor AWG
% (not actually from AWG but from the qctoolkit Tabor Driver instance)
%
% --- Outputs -------------------------------------------------------------
% seq_table				       : Cell of sequence tables for each Tabor
%                                channel pair
%
% --- Inputs --------------------------------------------------------------
% program_name				   : Program name for which sequence table is 
%                                returned
% advanced_seq_table_flag      : Get advanced sequence table if true.
%                                Default is false.
% awg_channel_pair_identifiers : Some substring in the channel pair 
%                                identifiers to be matched. Sequence tables
%                                are sorted in the same order as channel
%                                pair identifiers substrings passed in this
%																 variable. Default is {'AB', 'CD'}. 
% verbosity                    : Print sequence table to command line.
%                                Default is 0.
% return_python_list           : Returns a python list object instead of a
%                                matlab cell. This makes the function 
%                                faster as the conversion is slow.
%                                Dafault is false.
%
% -------------------------------------------------------------------------
% (c) 2018/06 Pascal Cerfontaine and Marcel Meyer
% (cerfontaine@physik.rwth-aachen.de)

global plsdata
hws = plsdata.awg.hardwareSetup;

if nargin < 2 || isempty(advanced_seq_table_flag)
	advanced_seq_table_flag = false;
end
if nargin < 3 || isempty(awg_channel_pair_identifiers)
	awg_channel_pair_identifiers = {'AB', 'CD'};
end	
if nargin < 4 || isempty(verbosity)
	verbosity = 0;
end	
if nargin < 5 || isempty(return_python_list)
  return_python_list = false;
end
if advanced_seq_table_flag
	seq_txt = 'A';
else
	seq_txt = '';
end

known_awgs = util.py.py2mat(hws.known_awgs);
sort_indices = cellfun(@(x)(find(  cellfun(@(y)(~isempty(strfind(char(x.identifier), y))), awg_channel_pair_identifiers)  )), known_awgs);
known_awgs = known_awgs(sort_indices);

for k = 1:length(known_awgs)
	known_programs{k} = util.py.py2mat(py.getattr(known_awgs{k}, '_known_programs'));

	if verbosity > 0
		util.disp_section(sprintf('%s %sST: %s', awg_channel_pair_identifiers{k}, seq_txt, program_name));
	end
	
	if isfield(known_programs{k}, program_name)
		tabor_program{k} = known_programs{k}.(program_name){2};

    if advanced_seq_table_flag
      seq_table{k} = py.getattr(tabor_program{k}, '_advanced_sequencer_table');
    else
      seq_table{k} = py.getattr(tabor_program{k}, '_sequencer_tables');
    end
    
    if verbosity > 0
      disp(seq_table{k});
    end
    
    if ~return_python_list
      seq_table{k} = util.py.py2mat(seq_table{k});
    end
	else
		tabor_program{k} = {};
		seq_table{k} = {};
		
		if verbosity > 0
			disp('  Program not present');
		end	
	end			
end	

if verbosity > 0
	fprintf('\n');
	util.disp_section();
end

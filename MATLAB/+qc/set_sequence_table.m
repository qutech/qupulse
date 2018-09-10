function set_sequence_table(program_name, seq_table, advanced_seq_table_flag, awg_channel_pair_identifiers, verbosity, input_python_list)
% SET_SEQUENCE_TABLE Manually override sequence table of program on Tabor AWG
%
% This only changes the sequence table in the associated Tabor channel
% pairs in qctoolkit. In order to actually update the sequence table on the
% AWG, you still need to run qc.change_armed_program(program_name, ...).
%
% --- Inputs --------------------------------------------------------------
% program_name						     : Program name for which sequence table is set
% seq_table                    : Cell of sequence table to set on each
%                                Tabor channel pair. Empty elements will
%                                not be set.
% advanced_seq_table_flag      : Set advanced sequence table if true.
%																 Default is false.
% awg_channel_pair_identifiers : Some substring in the channel pair 
%                                identifiers to be matched. Sequence tables
%                                are sorted in the same order as channel
%                                pair identifiers substrings passed in this
%																 variable. Default is {'AB', 'CD'}. 
% verbosity                    : Print sequence table to command line.
%																 Default is 0.
% -------------------------------------------------------------------------
% (c) 2018/06 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)
	
	global plsdata
	hws = plsdata.awg.hardwareSetup;

  if ~input_python_list
	seq_table = int_typecast(seq_table);
  end
  
	if nargin < 3 || isempty(advanced_seq_table_flag)
		advanced_seq_table_flag = false;
	end
	if nargin < 4 || isempty(awg_channel_pair_identifiers)
		awg_channel_pair_identifiers = {'AB', 'CD'};
	end	
	if nargin < 5 || isempty(verbosity)
		verbosity = 0;
  end	
  if nargin <6 || isempty(input_python_list)
    input_python_list = false;
  end

	known_awgs = util.py.py2mat(hws.known_awgs);
	sort_indices = cellfun(@(x)(find(  cellfun(@(y)(~isempty(strfind(char(x.identifier), y))), awg_channel_pair_identifiers)  )), known_awgs);
	known_awgs = known_awgs(sort_indices);

	assert(numel(seq_table) == length(known_awgs), 'Sequence table needs to be a cell with an element for each of the %i channel pairs.', length(known_awgs));

	for k = 1:numel(seq_table)	
		known_programs{k} = util.py.py2mat(py.getattr(known_awgs{k}, '_known_programs'));

		if isfield(known_programs{k}, program_name) && ~isempty(seq_table{k})
      if input_python_list
        if advanced_seq_table_flag
          known_awgs{k}.set_program_advanced_sequence_table(program_name, seq_table{k});
          % known_awgs{k}.set_program_advanced_sequence_table(program_name, seq_table{k});
        else
          known_awgs{k}.set_program_sequence_table(program_name, seq_table{k}); % Since it has to be a list inside a list, but this list if list is only trivial if advanced seq table is trivial, otherwiese each entry can be called by advanced seq table
          % known_awgs{k}.set_program_sequence_table(program_name,seq_table{k});
        end
      else
        if advanced_seq_table_flag
          known_awgs{k}.set_program_advanced_sequence_table(program_name, py.list(seq_table{k}));
          % known_awgs{k}.set_program_advanced_sequence_table(program_name, seq_table{k});
        else
          known_awgs{k}.set_program_sequence_table(program_name, py.list(seq_table{k})); % Since it has to be a list inside a list, but this list if list is only trivial if advanced seq table is trivial, otherwiese each entry can be called by advanced seq table
          % known_awgs{k}.set_program_sequence_table(program_name,seq_table{k});
        end
      end

		end			
	end	
	
	if verbosity > 0
		qc.get_sequence_table(program_name, advanced_seq_table_flag, awg_channel_pair_identifiers, verbosity);
	end

end
	

function out = int_typecast(in)
	
	if iscell(in)
		out = {};
		for k = 1:numel(in)
			out{k} = int_typecast(in{k});
		end	
	elseif ~isempty(in)
		out = int64(in);
	end
	
end
	
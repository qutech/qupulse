% function to display the sequence table hold by qctoolkit Tabor instance 
% or given in the varargins
% -------------------------------------------------------------------------
% Notes:
%   - if varargin.seq_table is empty the sequence table saved in the qctoolkit 
%   Tabor object is plotted -> function uses qc.get_sequence_table internaly
% -------------------------------------------------------------------------
% written by Marcel Meyer 08|2018


function disp_awg_seq_table(varargin)

  global plsdata

  defaultArgs = struct(...
    'seq_table', {{}}, ...
    'programName', plsdata.awg.currentProgam, ...
    'advancedSeqTableFlag', false ...
    );
  args = util.parse_varargin(varargin, defaultArgs);


  if isempty(args.seq_table)
    seq_table = qc.get_sequence_table(args.programName, args.advancedSeqTableFlag);
  else
    assert(iscell(args.seq_table), 'wrong format sequence table')
    seq_table = args.seq_table;
  end

  disp('   ');
  disp('[i] Table 1 is for channel pair AB and table 2 for channel pair CD.');
  disp('   ');

  counter = 0;
  tmpEntry = '';

  for k = 1:2
    if isempty(seq_table{k})
      warning('-- empty sequence table at channel nr %i -- \n', k);
    else
      if ~args.advancedSeqTableFlag

        fprintf('--- table %d -----------------\n', k);
        for n = 1:length(seq_table{k})
          fprintf(' -- sub table %d ---------\n', n);

          tmpEntry = seq_table{k}{n}{1};
          counter = 0;
          for i=1:length(seq_table{k}{n})
            if isequal(tmpEntry, seq_table{k}{n}{i})
              counter = counter+1;
            else

              fprintf('    rep = %d', counter);
              disp(tmpEntry);

              tmpEntry = seq_table{k}{n}{i};
              counter = 1;
            end
          end
          fprintf('    rep = %d', counter);
          disp(tmpEntry);
          disp('-----------------------------')
        end
        



      else

        fprintf('--- table %d -----------------\n', k);

        tmpEntry = seq_table{k}{1};
        counter = 0;
        for i=1:length(seq_table{k})
          if isequal(tmpEntry, seq_table{k}{i})
            counter = counter+1;
          else
            fprintf('    rep = %d', counter);
            disp(tmpEntry);

            tmpEntry = seq_table{k}{i};
            counter = 0;
          end
        end
        fprintf('    rep = %d', counter);
        disp(tmpEntry);
        disp('-----------------------------')

      end


    end
  end
end
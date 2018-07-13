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

  counter = 0;
  tmpEntry = '';

  for k = 1:2
    if isempty(seq_table{k})
      warning('-- empty sequence table or no program with this name --');
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
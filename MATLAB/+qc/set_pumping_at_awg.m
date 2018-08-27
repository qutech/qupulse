function set_pumping_at_awg(pumpingConfig, varargin)

global plsdata

defaultArgs = struct(...
  'programName', plsdata.awg.currentProgam, ...
  'turnOffAWG', true, ...
  'speedUp', false ...
  );
args = util.parse_varargin(varargin, defaultArgs);

if ~args.speedUp
  seqTable = qc.get_sequence_table(args.programName, false);
else
  seqTable = qc.get_sequence_table(args.programName, false, {'AB', 'CD'}, false, true);
end
seqTableCheck = true;
report = '';
pumpSubTab = seqTable{1}{end};





%---------------- some checks ---------------------------------------------
if ~args.speedUp
  
  %check if there are six entries for the three pumping types for each qubit
  if length(pumpSubTab) < 6
    seqTableCheck = false;
    report = ' -- There are not six waveforms at the end of the sequence table! They might be put together or not uploaded or not at the end of the sequence table. -- ';
  end
  
  %test if every waveform is different/has a different
  if seqTableCheck
    for i = 0:5
      for j = 0:5
        if (i~=j) && (pumpSubTab{end-i}{2} == pumpSubTab{end-j}{2})
          report = ' -- Not all waveforms for pumping (that are assumed to be different) are different to each other! -- ';
          seqTableCheck = false;
        end
      end
    end
  end
  
  %test if both channel pairs have the same pumping sequence table part
  for i = 1:6
    if seqTableCheck && ~isequal(seqTable{1}{end}{end-i+1}, seqTable{2}{end}{end-i+1})
      report = ' -- Not the same pumping configuration on both channel pairs of the AWG! -- ';
      seqTableCheck = false;
    end
  end
  
end


%------------- reading out the pumping configuration ----------------------

if ~seqTableCheck
  warning(report);
else
  if ~args.speedUp
    seqTable{1}{end}{end-5}{1} = pumpingConfig.n_s_AB;
    seqTable{1}{end}{end-4}{1} = pumpingConfig.n_t_AB;
    seqTable{1}{end}{end-3}{1} = pumpingConfig.n_cs_AB;
    seqTable{1}{end}{end-2}{1} = pumpingConfig.n_s_CD;
    seqTable{1}{end}{end-1}{1} = pumpingConfig.n_t_CD;
    seqTable{1}{end}{end-0}{1} = pumpingConfig.n_cs_CD;
    seqTable{2}{end}{end-5}{1} = pumpingConfig.n_s_AB;
    seqTable{2}{end}{end-4}{1} = pumpingConfig.n_t_AB;
    seqTable{2}{end}{end-3}{1} = pumpingConfig.n_cs_AB;
    seqTable{2}{end}{end-2}{1} = pumpingConfig.n_s_CD;
    seqTable{2}{end}{end-1}{1} = pumpingConfig.n_t_CD;
    seqTable{2}{end}{end-0}{1} = pumpingConfig.n_cs_CD;
  else
    seqTable{1}{end} = py.list(seqTable{1}{end});
    seqTable{2}{end} = py.list(seqTable{1}{end});
    for i = 0:5
      seqTable{1}{end}{end-i} = py.list(seqTable{1}{end}{end-i});
      seqTable{2}{end}{end-i} = py.list(seqTable{2}{end}{end-i});
    end
    seqTable{1}{end}{end-5}{1} = py.int(pumpingConfig.n_s_AB);
    seqTable{1}{end}{end-4}{1} = py.int(pumpingConfig.n_t_AB);
    seqTable{1}{end}{end-3}{1} = py.int(pumpingConfig.n_cs_AB);
    seqTable{1}{end}{end-2}{1} = py.int(pumpingConfig.n_s_CD);
    seqTable{1}{end}{end-1}{1} = py.int(pumpingConfig.n_t_CD);
    seqTable{1}{end}{end-0}{1} = py.int(pumpingConfig.n_cs_CD);
    seqTable{2}{end}{end-5}{1} = py.int(pumpingConfig.n_s_AB);
    seqTable{2}{end}{end-4}{1} = py.int(pumpingConfig.n_t_AB);
    seqTable{2}{end}{end-3}{1} = py.int(pumpingConfig.n_cs_AB);
    seqTable{2}{end}{end-2}{1} = py.int(pumpingConfig.n_s_CD);
    seqTable{2}{end}{end-1}{1} = py.int(pumpingConfig.n_t_CD);
    seqTable{2}{end}{end-0}{1} = py.int(pumpingConfig.n_cs_CD);
  end
end

if args.speedUp
  qc.set_sequence_table(args.programName, seqTable, false, {'AB', 'CD'}, false, true);
else
  qc.set_sequence_table(args.programName, seqTable, false);
end
qc.change_armed_program(args.programName, args.turnOffAWG);

if args.turnOffAWG
  disp('turned AWG off');
end

end
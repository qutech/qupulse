function set_pumping_at_awg(pumpingConfig, varargin)

global plsdata

defaultArgs = struct(...
  'programName', plsdata.awg.currentProgam, ...
  'turnOffAWG', true ...
  );
args = util.parse_varargin(varargin, defaultArgs);

seqTable = qc.get_sequence_table(args.programName, false);
seqTableCheck = true;
report = '';
pumpSubTab = seqTable{1}{end};





%---------------- some checks ---------------------------------------------

%check if there are six entries for the three pumping types for each qubit
if length(pumpSubTab) ~= 6
  seqTableCheck = false;
  report = ' -- There are not six waveforms at the end of the sequence table! They might be put together or not uploaded or not at the end of the sequence table. -- ';
end

%test if every waveform is different/has a different 
if seqTableCheck
  for i = 1:6
    for j = 1:6
      if (i~=j) && (pumpSubTab{i}{2} == pumpSubTab{j}{2})
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




%------------- reading out the pumping configuration ----------------------

if ~seqTableCheck
  warning(report);
else
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
end

qc.set_sequence_table(args.programName, seqTable, false);
qc.change_armed_program(args.programName, args.turnOffAWG);

if args.turnOffAWG
  disp('turned AWG off');
end

end
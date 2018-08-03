function pumpingConfig = get_pumping_from_awg(varargin)

global plsdata

defaultArgs = struct(...
  'programName', plsdata.awg.currentProgam ...
  );
args = util.parse_varargin(varargin, defaultArgs);

pumpingConfig = struct();

seqTable = qc.get_sequence_table(args.programName, false);
seqTableCheck = true;
report = '';
pumpSubTab = seqTable{1}{end};





%---------------- some checks ---------------------------------------------

%check if there are six entries for the three pumping types for each qubit
if length(pumpSubTab) ~= 6
  seqTableCheck = false;
  report = ' -- There are not six waveforms at the end of the sequence table! They might be put together or not uploaded or not at the end of the sequence table. -- '
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
if seqTableCheck && ~isequal(seqTable{1}{end}, seqTable{2}{end})
  report = ' -- Not the same pumping configuration on both channel pairs of the AWG! -- ';
  seqTableCheck = false;
end




%------------- reading out the pumping configuration ----------------------

if ~seqTableCheck
  warning(report);
else
  pumpingConfig.n_s_AB  = pumpSubTab{end-5}{1};
  pumpingConfig.n_t_AB  = pumpSubTab{end-4}{1};
  pumpingConfig.n_cs_AB = pumpSubTab{end-3}{1};
  pumpingConfig.n_s_CD  = pumpSubTab{end-2}{1};
  pumpingConfig.n_t_CD  = pumpSubTab{end-1}{1};
  pumpingConfig.n_cs_CD = pumpSubTab{end-0}{1};
end




end
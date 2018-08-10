function seq_table = get_awg_seq_table(varargin)

	global plsdata

  defaultArgs = struct(...
    'programName', plsdata.awg.currentProgam, ...
    'advancedSeqTableFlag', false ...
    );
  args = util.parse_varargin(varargin, defaultArgs);

  seq_table = qc.get_sequence_table(args.programName, args.advancedSeqTableFlag);

end
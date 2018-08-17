function [wf1, wf2] = get_segment_waveform(program_name, channel_pair_index, memory_index, awg_channel_pair_identifiers)

global plsdata
  hws = plsdata.awg.hardwareSetup;

  if nargin < 4 || isempty(awg_channel_pair_identifiers)
    awg_channel_pair_identifiers = {'AB', 'CD'};
  end
  
  known_awgs = util.py.py2mat(hws.known_awgs);
  sort_indices = cellfun(@(x)(find(  cellfun(@(y)(~isempty(strfind(char(x.identifier), y))), awg_channel_pair_identifiers)  )), known_awgs);
  known_awgs = known_awgs(sort_indices);
  
  %one has to arm the program to access the plottableProgram object of the
  %program
  known_awgs{channel_pair_index}.arm(program_name);
  
  plottableProgram = known_awgs{channel_pair_index}.read_complete_program();
  
  wf1 = plottableProgram.get_segment_waveform(uint8(0), uint8(memory_index));
  wf2 = plottableProgram.get_segment_waveform(uint8(1), uint8(memory_index));
  
  wf1 = util.py.py2mat(wf1);
  wf2 = util.py.py2mat(wf2);
  
  wf1 = cell2mat(wf1);
  wf2 = cell2mat(wf2);
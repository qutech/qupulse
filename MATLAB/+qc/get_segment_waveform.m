function [wf1, wf2] = get_segment_waveform(program_name, channel_pair_index, memory_index, awg_channel_pair_identifiers)
% Get Wafeform of Sequencer Table Element
% PLEASE NOTE: works only for the Tabor AWG SIMULATOR
% PLEASE NOTE: program gets armed by calling this function
%
% --- Outputs -------------------------------------------------------------
% wf1                          : first channel y-values of AWG channelpair
% wf2                          : second channel y-values of AWG channelpair
%
% --- Inputs --------------------------------------------------------------
% program_name						     : Program name for which wafeform is 
%                                returned
% channel_pair_index           : 1 for channelpair AB and 2 for channelpair
%                                CD. Also see awg_channel_pair_identifier
%                                input
% memory_index                 : identifier number of element at the Tabor 
%                                AWG (corresponds to second column in 
%                                Sequencer Table 
% awg_channel_pair_identifiers : Some substring in the channel pair 
%                                identifiers to be matched. Sequence tables
%                                are sorted in the same order as channel
%                                pair identifiers substrings passed in this
%																 variable. Default is {'AB', 'CD'}.
%
% -------------------------------------------------------------------------
% 2018/08 Marcel Meyer
% (marcel.meyer1@rwth-aachen.de)

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
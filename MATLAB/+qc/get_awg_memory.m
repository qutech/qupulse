% function to get waveforms and sequence tables from the AWG
% -------------------------------------------------------------------------
% Notes:
%   - the function only works with the Tabor AWG Simulator not on the real
%   Tabor AWG
%   - the function arms the program that is inspected
% -------------------------------------------------------------------------
% written by Marcel Meyer 08|2018

function awg_memory_struct = get_awg_memory(program_name, awg_channel_pair_identifier)

  global plsdata
  
  assert(ischar(program_name), 'first argument of get_awg_memory must be string');
  
  if nargin < 2 || isempty(awg_channel_pair_identifier)
		awg_channel_pair_identifier = 'AB';
  else
    assert(ischar(awg_channel_pair_identifier), 'second argument of get_awg_memory must be string');
  end	
  
  % get AWG channelpair python object
  hws = plsdata.awg.hardwareSetup;
  known_awgs = util.py.py2mat(hws.known_awgs);
  sort_indices = cellfun(@(x)(~isempty(strfind(char(x.identifier), awg_channel_pair_identifier))), known_awgs);
  channelpair = known_awgs(find(sort_indices));
  channelpair = channelpair{1};
  
  % arm program at AWG
  try
    channelpair.arm(program_name);
  catch err
    warning('program seems not to be on AWG, upload it first, returning without returning memory');
    warning(err.message);
    return
  end
  
  % get a plottable program object -> qctoolkit Tabor driver gets sequence
  % tables and waveforms from the simulator
  plottableProgram = channelpair.read_complete_program();
  
  awg_memory_struct = util.py.py2mat(plottableProgram.to_builtin());

end
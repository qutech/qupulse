function change_armed_program(program_name, turn_awg_off)
% CHANGE_ARMED_PROGRAM Force arming of a program on Tabor AWG
% This function calls change_armed_program and each Tabor channel pair
% which contains the indicated program. The program needs to be already
% present on the AWG.
% --- Inputs --------------------------------------------------------------
% program_name						     : Program name which is armed
% turn_awg_off                 : Turn AWG off after arming the program.
%																 Default is true.
% -------------------------------------------------------------------------
% (c) 2018/06 Pascal Cerfontaine (cerfontaine@physik.rwth-aachen.de)

global plsdata
hws = plsdata.awg.hardwareSetup;

if nargin < 2 || isempty(turn_awg_off)
	turn_awg_off = true;
end

known_awgs = util.py.py2mat(hws.known_awgs);

for k = 1:length(known_awgs)
	known_programs{k} = util.py.py2mat(py.getattr(known_awgs{k}, '_known_programs'));

	if isfield(known_programs{k}, program_name)
		known_awgs{k}.change_armed_program(program_name);
	end			
end	

if turn_awg_off
	awgctrl('off');
end
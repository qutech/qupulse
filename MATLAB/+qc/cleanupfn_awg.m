function scan = cleanupfn_awg(scan)
	
	if nargin < 1
		scan = [];
	end
	
	evalin('caller', 'cleanupFnAwg = onCleanup(@()({awgctrl(''off''), fprintf(''Executing cleanup function: Turned AWG outputs off\n'')}));');
% 	evalin('caller', 'cleanupFnRfMsg = onCleanup(@()(fprintf(''Executing cleanup function: Disarmed AWG\n'')));');
% 	evalin('caller', 'cleanupFnRf1 = onCleanup(@()(disarm_program(7)));');
	
end

function arm_and_sync(awg, program_name)
awg.arm(program_name)
ziDAQ('sync');

end

function disarm_program(inst_index)
global smdata;

arm_and_sync(smdata.inst(inst_index).data.qupulse.channel_tuples{1}, py.None);

end
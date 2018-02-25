function plot_tabor_pulse(awg)

program = awg.read_complete_program();

wfs = util.py.py2mat(program.get_waveforms());
reps = util.py.py2mat(program.get_repetitions());
n_wfs = numel(wfs);

f = figure;



tabgroup = uitabgroup(mainfig, 'Position', [.05 .1 .9 .8]);

for k = 1:n_wfs
    tab(k)=uitab(tabgroup,'Title', sprintf('Wf_%i', k));
	
    axes('parent',tab(k))
    
	plot(wfs{k});
	
	legend(sprintf('%i times', reps(k)));
end
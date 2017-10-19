function plot_pulse(pulse, parameters, npoints)
	%%
sequencer = py.qctoolkit.pulses.Sequencer();

kwargs = pyargs('parameters', parameters);

sequencer.push(pulse, kwargs);

instantiated_pulse = sequencer.build();

pulse_duration_in_s = qctoolkit.get_pulse_duration(pulse, parameters);
pulse_duration_in_ns = pulse_duration_in_s * 1e9;
%%
if nargin < 3
	npoints = 100;
end
%%
sample_rate = npoints / pulse_duration_in_ns;
%%
data = util.py.py2mat(py.qctoolkit.pulses.plotting.render(instantiated_pulse, pyargs('sample_rate', sample_rate)));

t = data{1};
figure;
hold on

for chan_name=fieldnames(data{2})'
	plot(t, data{2}.(chan_name{1}));
end

legend(fieldnames(data{2})');
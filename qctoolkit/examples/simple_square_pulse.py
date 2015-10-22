from qctoolkit.pulses.table_pulse_template import TablePulseTemplate
from qctoolkit.pulses.sequence_pulse_template import SequencePulseTemplate
from qctoolkit.pulses.plotting import Plotter, plot
from qctoolkit.pulses.sequencing import Sequencer
from matplotlib import pyplot as plt

#
#               c
#              /|
#             / |
#            /  |
#           /   |
#          b    |
#          |    |
#          |    |
#          |    |
# a--------|    ------------------d
#
# Point: Tuple
# a: (0,0)
# b: ('t_up', 'value1')
# c: ('t_down', 'value2')
# d: ('end', 0)
# t_down should not be given by the user, instead give the time between c and b as 'length'

squarePulse = TablePulseTemplate() # Prepare new empty Pulse
# Then add pulses sequentially
squarePulse.add_entry('t_up', 'value1', interpolation='hold') # hold is the standard interpolation value
squarePulse.add_entry('t_down', 'value2', interpolation='linear')
squarePulse.add_entry('end', 0, interpolation='jump')

# We can just plug in values for the parameters to get an actual pulse:
parameters = {'t_up': 200,
              't_down': 2000,
              'end': 4000,
              'value1': 2.2,
              'value2': 3.0}

# with these parameters, we can plot the pulse:
plot(squarePulse, parameters)

# To re-parametrize we can simply wrap the pulse definition in a SequencePulseTemplate that provides functionality
# for mapping its own parameters onto children parameters.

mapping = {}
mapping['t_up']   = 'start'
mapping['t_down'] = 'start + length'
mapping['value1'] = 'value1'
mapping['value2'] = 'value2'
mapping['end']    = 'pulse_length * 0.5'

doubleSquare = SequencePulseTemplate([(squarePulse, mapping),
                                      (squarePulse, mapping)], # dictionaries with mapping functions from external parameters to subtemplate parameters
                                     ['start', 'length', 'value1', 'value2', 'pulse_length']) # declare the new template's external parameters

params = dict(start=5,
              length=20,
              value1=10,
              value2=15,
              pulse_length=500)

plot(doubleSquare, params)

nested_mapping = dict(start='start',
                      length='length',
                      value1='10',
                      value2='20',
                      pulse_length='pulse_length * 0.5')

nested_pulse = SequencePulseTemplate([(doubleSquare, nested_mapping),
                                      (doubleSquare, nested_mapping)],
                                     ['start', 'length', 'pulse_length'])

params2 = dict(start=10, length=100, pulse_length=1000)

# # Instead of calling the convenience plot function, we can also use the PlottingSequencer directly
# # This is also an instructive example on how to use sequencers.
# plotter = Plotter()
# sequencer = Sequencer(plotter)
# sequencer.push(nested_pulse, params2)
# times, voltages = plotter.render()
# plt.step(times, voltages)
# plt.show() # eh voila, a sequence of four pulses

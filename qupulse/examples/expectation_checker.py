import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from typing import Dict, Tuple, Optional
import warnings
import time
from datetime import datetime
import xarray as xr

from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.program import waveforms
from qupulse.program.linspace import LinSpaceBuilder, to_increment_commands
from qupulse.program.loop import LoopBuilder
from qupulse.hardware.setup import HardwareSetup, PlaybackChannel, MarkerChannel, MeasurementMask
from qupulse.hardware.dacs import AlazarCard
from qupulse.pulses import PointPT, ConstantPT, RepetitionPT, ForLoopPT, TablePT, \
    FunctionPT, AtomicMultiChannelPT, SequencePT, MappingPT, ParallelConstantChannelPT
from qupulse.program import ProgramBuilder
from qupulse.plotting import plot, render, _render_loop

from qupulse_hdawg.zihdawg import HDAWGRepresentation, HDAWGChannelGrouping

import atsaverage
from atsaverage import alazar
from atsaverage.config import ScanlineConfiguration, CaptureClockConfiguration, EngineTriggerConfiguration,\
    TRIGInputConfiguration, InputConfiguration
import atsaverage.server
import atsaverage.client
import atsaverage.core
import atsaverage.config as config


#%%

class HDAWGAlazar():
    '''
    gets locally installed alazar and hdawg in hardcoded config
    meaning 2V-peak-to-peak range, channel name scheme "ZI0_A", "ZI0_B" etc.
    '''
    def __init__(self,
                 awg_serial: str = "DEVXXXX",
                 awg_interface: str = "USB", #or 1GbE for LAN
                 awg_sample_rate: float = 1e9,
                 
                 ):
        
        self._hw_setup = HardwareSetup()
        self._alazar = get_alazar()
        self._init_alazar()
        self._awg_serial = awg_serial
        self._awg_sample_rate = awg_sample_rate
        self._hdawg = get_hdawg(self._hw_setup,awg_serial,awg_interface)
        self._init_hdawg()
        
    @property
    def alazar(self) -> AlazarCard:
        return self._alazar
        
    @property
    def hdawg(self) -> HDAWGRepresentation:
        return self._alazar
    
    def _init_hdawg(self):
        self._hdawg.api_session.setDouble(f'/{self._awg_serial}/system/clocks/sampleclock/freq', self._awg_sample_rate)

        for i in range(8):
            self._hdawg.api_session.setDouble(f'/{self._awg_serial}/sigouts/{i}/range', 2)
            self._hdawg.api_session.setInt(f'/{self._awg_serial}/sigouts/{i}/on', 1)

        #Hacky piece of code to enable Hardware triggering on the second AWG, will probably be deprecated at some point
        for awg in self._hw_setup.known_awgs:
            # awg._program_manager._compiler_settings[0][1]['trigger_wait_code']='waitDigTrigger(1);'
            awg._program_manager._compiler_settings[0][1]['trigger_wait_code']=''
    
    def _init_alazar(self):
        
        atsaverage.server.Server.default_instance.start(b'blabla')
        atsaverage.server.Server.default_instance.connect_gui_window_from_new_process()
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        
    
class ExpectationChecker():
    '''
    trig-channels are all marker outputs from hdawg.
    ensure more or less equal length cables from hdawg to alazar (including trig)
    assumes channels in ascending order, e.g. first alazar channel to first hdawg
    '''
    MARKER_CHANS = {f'ZI0_{x}_MARKER_FRONT' for x in 'ABCDEFGH'}
    PLAY_CHANS = {f'ZI0_{x}' for x in 'ABCDEFGH'}
    ALLOWED_CHANS = MARKER_CHANS | PLAY_CHANS
    MEAS_CHANS = 'ABCD'
    ALAZAR_RAW_RATE = 100_000_000
    MEAS_NAME = 'prog'
    
    def __init__(self,
                 devices: HDAWGAlazar|None,
                 pt: PulseTemplate|None,
                 program_builder: ProgramBuilder|None,
                 sample_rate_alazar: float = 1e-1, #ns
                 sample_rate_pt_plot: float = 1e0, #ns
                 save_path: str|None = None,
                 data_offsets: Dict[str,float] = {'t_offset':100,'v_offset':0.008,'v_scale':0.9975}
                 ):
        
        self._devices = devices
        if pt is not None:
            self._pt = self._approve_pt(pt)
        else:
            self._pt = None
        self.program_builder = program_builder
        
        self._number_samples_per_average = int(sample_rate_alazar*1e9 // self.ALAZAR_RAW_RATE)
        assert sample_rate_alazar*1e9 % self.ALAZAR_RAW_RATE == 0, f'only sample rates as multiples of {1/self.ALAZAR_RAW_RATE*1e9} ns'
        assert self._number_samples_per_average >= 1, 'alazar sample rate too high'
        
        self._save_path = save_path
        
        self._data_offsets = data_offsets
        
    @classmethod
    def from_file(cls, path: str, data_offsets: Optional[Dict[str,float]] = None) -> 'ExpectationChecker':
        ds = xr.open_dataset(path)
        file_data_offsets = {
            't_offset': ds.attrs['offsets_to_be_applied'][0],
            'v_offset': ds.attrs['offsets_to_be_applied'][1],
            'v_scale': ds.attrs['offsets_to_be_applied'][2]
            }
        
        
        ec = cls(None,None,None,save_path=None,data_offsets=data_offsets if data_offsets is not None else file_data_offsets)
        ec._result = ds
        return ec
    
    def _approve_pt(self, pt: PulseTemplate) -> bool:
        
        assert pt.defined_channels <= self.ALLOWED_CHANS, 'name chans according to hardcoded naming scheme (ZI0_X)'
        
        if any(ch in self.MARKER_CHANS for ch in pt.defined_channels):
            print('overwriting marker channel(s) to be always on')
        pt_mapped = ParallelConstantChannelPT(pt, {x:1. for x in self.MARKER_CHANS}, identifier=pt.identifier)
        
        
        return pt_mapped
    
    def run(self):
        self._register_program()
        self._run_aqcuire()
        self._save()
        self._plot_result()
    
    def _save(self):
        if self._save_path:
            name = self._pt.identifier+'_' if self._pt.identifier else ''
            name += str(datetime.now())[:19].replace(':','_').replace(' ','_')
            self._result.to_netcdf(self._save_path+'//'+name+'.nc')
    
    def _plot_result(self):
        
        fig = plt.figure(figsize=(5,3),dpi=300)
        gs0 = fig.add_gridspec(1, 1, height_ratios=[1.0,])
        ax = fig.add_subplot(gs0[0])

        SMALL_SIZE = 8
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 12
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-1)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rc("xtick.minor", visible=True)
        plt.rc("ytick.minor", visible=True)
        
        colors = plt.cm.Dark2.colors
        
        
        #measured:
        for ch in [ch for ch in self._result.data_vars if ch.endswith('meas')]:
            ax.plot(self._result[ch].coords['time'].data+self._data_offsets.get('t_offset',0.),
                    self._data_offsets.get('v_scale',1.)*self._result[ch].data+self._data_offsets.get('v_offset',0.),
                    color=colors[self.MEAS_CHANS.find(ch[0])],
                    linestyle='',
                    marker="o",markerfacecolor=(1.,1.,1.,0.),#colors[i]+(0.5,),
                    markeredgecolor=colors[self.MEAS_CHANS.find(ch[0])]+(0.5,),markeredgewidth=.3,markersize=2.5,
                    # marker
                    label=ch
                    )
        
        #expectation:
        for ch in [ch for ch in self._result.data_vars if ch.endswith('exp')]:
            ax.plot(self._result[ch].coords['time'].data,self._result[ch].data,
                    color=colors[self.MEAS_CHANS.find(ch[0])],
                    linestyle='-',
                    linewidth=1.,
                    path_effects=[pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()],
                    zorder=2,
                    # marker="o",markerfacecolor=None,
                    # markeredgecolor=colors[i],markeredgewidth=.6,markersize=1.,
                    label=ch
                    )
            
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Voltage (V)')
        
        if n:=self._result.attrs['name']:
            ax.set_title(n)
        
        ax.legend(ncols=2)
        
        fig.tight_layout()
        
        return fig, ax


    def _examine_diff(self):
        pass
    
    def _register_program(self):
        
        self._devices._hw_setup.clear_programs()
        
        def run_callback(): 
            awgs = []
            for a in self._devices._hw_setup.known_awgs:
                if a.__class__.__name__ != 'TaborChannelPair': #TODO: not negative check (but zihdawg channels potentially have different names)
                    awgs += [a]
            for awg in awgs[::]:
                awg.enable(True)
                
            return
        
        name = self.MEAS_NAME
        for i,meas_chan in enumerate(self.MEAS_CHANS):
           self._devices.alazar.register_mask_for_channel(name+meas_chan,i)
       
        operations = []
        measurement_masks = []
        for meas_chan in self.MEAS_CHANS:
            mask_name = name + meas_chan
            operations.append(atsaverage.operations.ChunkedAverage(
                maskID=mask_name, identifier=mask_name, chunkSize=self._number_samples_per_average
                ))
            measurement_masks.append(MeasurementMask(self._devices.alazar, mask_name))
        self._devices._hw_setup.set_measurement(name,
                                                measurement_masks,
                                                allow_multiple_registration=True)
        self._devices.alazar.register_operations(name,operations)
        
        prog = self._pt.create_program(program_builder=self.program_builder,)
        # measurements = {mw_name: (begin_length_list[:, 0], begin_length_list[:, 1])}
        # measurements = {name+meas_chan: ([0.,],[float(self._pt.duration),]) for meas_chan in self.MEAS_CHANS}
        measurements = {name: ([0.,],[float(self._pt.duration),])}

        self._devices._hw_setup.register_program(self.MEAS_NAME, prog, run_callback=run_callback,
                                                 measurements=measurements
                                                 )
        
        
    def _run_aqcuire(self):
        
        _results_dict = alazar_measure(self._devices,program_id=self.MEAS_NAME)
        
        dur = float(self._pt.duration)
        times = np.linspace(0.,dur,len(_results_dict['progA'][0]))

        _, voltages, _ = render_pt_with_times(self._pt,times,{ch for ch in self.PLAY_CHANS})

        self._result = xr.Dataset(
                        {
                            key.removeprefix(self.MEAS_NAME)+'_meas': (["time",], arrarr[0]) 
                            for key,arrarr in _results_dict.items() 
                            }|
                        {
                            key.removeprefix('ZI0_')+'_exp': (["time",], val) for key,val in voltages.items()
                            }
                        ,
                        coords={
                            "time": times,
                        },
                        attrs={'name': self._pt.identifier if self._pt.identifier else '',
                               'offsets_to_be_applied': (self._data_offsets.get('t_offset',0.),
                                                         self._data_offsets.get('v_offset',0.),
                                                         self._data_offsets.get('v_scale',1.))
                               }
                    )
        
        return


#%%


def render_pt_with_times(pt,times,plot_channels
            ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    
    waveform, measurements = _render_loop(pt.create_program(), render_measurements=False)
    
    if waveform is None:
        return np.array([]), dict(), measurements
    
    channels = waveform.defined_channels

    voltages = {ch: waveforms._ALLOCATION_FUNCTION(times, **waveforms._ALLOCATION_FUNCTION_KWARGS)
                for ch in channels if ch in plot_channels}
    # print(voltages.keys())
    # print(plot_channels)
    for ch, ch_voltage in voltages.items():
        waveform.get_sampled(channel=ch, sample_times=times, output_array=ch_voltage)

    return times, voltages, measurements
  
    
#%% measure program

def alazar_measure(devices: HDAWGAlazar, program_id: str = None,
                   *args, **kwargs,
                   ) -> Dict[str,np.ndarray]:
    
    # alazar_hack_kwargs = {}
    #Error 582 "ApiBufferOverflow":
    alazar_hack_kwargs = {'extend_to_all_channels':True}
    
    try:
        devices._hw_setup.run_program(program_id,
                                      # **alazar_hack_kwargs
                                      )
        print('\n--- Program run ---')
        results = devices.alazar.measure_program()
        print('--- Program measured ---')

    except RuntimeError as err:
        print(err)
        print('Error occured, second run')
        # pass
        time.sleep(2.0)
        devices.alazar.update_settings=True
        devices._hw_setup.run_program(program_id,
                                      # **alazar_hack_kwargs
                                      )
        results = devices.alazar.measure_program()   
        
    #dict with keys single_measurement_name+[A/B/C/D] for channels,
    #of length corresponding to
    #some datapoints for every measurement window
    
    if 'average_n_consecutive' in kwargs.keys():
        average_n_consecutive = kwargs['average_n_consecutive']
        for key, res_arr in results.items():
            assert len(res_arr)%average_n_consecutive==0,'should have been divisible by average_n_consecutive'
            reshaped_arr = res_arr.reshape(-1, average_n_consecutive)
            results[key] = reshaped_arr.mean(axis=1)
    
    return results


#%% setup alazar

def get_alazar(masks=None,operations=None,number_of_samples=None):
    
    r = 2.5
    rid = alazar.TriggerRangeID.etr_TTL

    # trig_level = int((r + 0.15) / (2*r) * 255)
    # I don't know if this is correct?
    
    trig_level = 4
    
    assert 0 <= trig_level < 256
    
    
    
    
    config = ScanlineConfiguration()
    config.triggerInputConfiguration = TRIGInputConfiguration(triggerRange=rid)
    config.triggerConfiguration = EngineTriggerConfiguration(triggerOperation=alazar.TriggerOperation.J,
                                                             triggerEngine1=alazar.TriggerEngine.J,
                                                             triggerSource1=alazar.TriggerSource.external,
                                                             triggerSlope1=alazar.TriggerSlope.positive,
                                                             triggerLevel1=trig_level,
                                                             triggerEngine2=alazar.TriggerEngine.K,
                                                             triggerSource2=alazar.TriggerSource.disable,
                                                             triggerSlope2=alazar.TriggerSlope.positive,
                                                             triggerLevel2=trig_level)
    
    config.captureClockConfiguration = CaptureClockConfiguration(
        # source=alazar.CaptureClockType.external_clock,
        # alazar.CaptureClockType.fast_external_clock,
        source=alazar.CaptureClockType.external_clock_10MHz_ref,
        # source=alazar.CaptureClockType.internal_clock,
        # source=alazar.CaptureClockType.slow_external_clock,
        # samplerate=alazar.SampleRateID.rate_100MSPS,
        # ATS API:
        # "If the clock source chosen is INTERNAL_CLOCK, this value is a member
        # of ALAZAR_SAMPLE_RATES that defines the internal sample rate to choose.
        # Valid values for each board vary. If the clock source chosen is
        # EXTERNAL_CLOCK_10MHZ_REF, pass the value of the sample clock to
        # generate from the reference in hertz. The values that can be generated
        # depend on the board model. Otherwise, the clock source is external,
        # pass SAMPLE_RATE_USER_DEF to this parameter." 
        # So:
        samplerate=100_000_000, #for 100MSPS
        
        # samplerate=alazar.SampleRateID.rate_125MSPS,

        # decimation=1,
        # samplerate=alazar.SampleRateID.user_def
        )
    
    config.inputConfiguration = 4*[InputConfiguration(input_range=alazar.InputRangeID.range_2_V)]
    # config.totalRecordSize = 0
    # config.aimedBufferSize = 10*config.aimedBufferSize

    # is set automatically
    assert config.totalRecordSize == 0
    
    config.autoExtendTotalRecordSize = 0
    config.AUTO_EXTEND_TOTAL_RECORD_SIZE = 0 #what does th is even do?
    
    #test from m.b.'s example scripts
    if masks is not None and operations is not None and number_of_samples is not None:
        print('masks & operations set')
        config.masks = masks
        config.operations = operations
        config.totalRecordSize = number_of_samples
        config.autoExtendTotalRecordSize = 1
    
    print(config)
    
    alazar_DAC = AlazarCard(atsaverage.core.getLocalCard(1, 1),config,)
    # alazar_DAC.card.applyConfiguration(config)
    # alazar_DAC.card.apply_board_configuration(config)
    #alazar_DAC.card.
    alazar_DAC.card.triggerTimeout = 1200000
    alazar_DAC.card.acquisitionTimeout = 12000000
    alazar_DAC.card.computationTimeout = 1200000
    
    #alazar_DAC.config.rawDataMask =atsaverage.atsaverage.ChannelMask(0)
    
    # channels=['A','B','C','D']
    # for i,channel in enumerate(channels):
    #      alazar_DAC.register_mask_for_channel(channel, i)

    # return alazar_DAC, config
    return(alazar_DAC)


#%% get hdawg



def get_hdawg(hardware_setup: HardwareSetup, Serial='DEVXXXX', device_interface="USB", address='localhost',idx=None):
    if idx is None:
        idx=str(len(hardware_setup.known_awgs))
    hdawg = HDAWGRepresentation(Serial, device_interface=device_interface, data_server_addr=address)
    
    hdawg.reset()
    
    hdawg.channel_grouping = HDAWGChannelGrouping.CHAN_GROUP_1x8
    
    CH_NAMES = 'ABCDEFGH'
    
    for awg in hdawg.channel_tuples:
        n_channels = awg.num_channels
        awg_channels = CH_NAMES[:n_channels]
        CH_NAMES = CH_NAMES[n_channels:]
        
        for ch_i, ch_name in enumerate(awg_channels):
            playback_name = 'ZI'+str(idx)+'_{ch_name}'.format(ch_name=ch_name)
            hardware_setup.set_channel(playback_name, PlaybackChannel(awg, ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_FRONT', MarkerChannel(awg, 2 * ch_i))
            hardware_setup.set_channel(playback_name + '_MARKER_BACK', MarkerChannel(awg, 2 * ch_i + 1))
    
    return hdawg



from qupulse.hardware.awgs_new_driver.features import SCPI, ProgramManagement, StatusTable
from qupulse.hardware.awgs_new_driver.tabor import TaborDevice, TaborVoltageRange

testDevice = TaborDevice("testDevice",
                         "127.0.0.1",
                         reset=True,
                         paranoia_level=2)

# print(testDevice.dev_properties)

# testDevice.send_cmd(':INST:SEL 1; :OUTP:COUP DC')
# testDevice.send_cmd(':VOLT 0.7')

# print(testDevice._is_coupled())


print(testDevice._is_coupled())

testDevice.channel_tuples[0][ProgramManagement]._cont_repetition_mode()
testDevice.channel_tuples[1][ProgramManagement]._cont_repetition_mode()
print(testDevice[StatusTable].get_status_table())
#testDevice.channel_tuples[1][ProgramManagement]._trig_repetition_mode()
print(testDevice[StatusTable].get_status_table())

testDevice[SCPI].send_cmd(':INST:SEL 1')
testDevice[SCPI].send_cmd(':TRIG')



#testDevice[SCPI].send_cmd(':INST:COUP:STAT ON')

print(testDevice._is_coupled())

print(testDevice.channels[0][TaborVoltageRange].amplitude)
print(testDevice.channels[0][TaborVoltageRange].offset)

#print(testDevice.channel_tuples[0].read_sequence_tables)
#print(testDevice.channel_tuples[0].read_waveforms)
#testDevice[SCPI].send_cmd('init:cont 0')
#testDevice.cleanup()



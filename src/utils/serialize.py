__author__ = 'bethke'

from src.pulses.TablePulseTemplate import TablePulseTemplate

import json
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj.to_json: # to_json method exists:
            return json.JSONEncoder.default(self,obj.to_json)
        else: # use default encoder for everything else
            return json.JSONEncoder.default(self,obj)

#class CustomDecoder(json.JSONDecoder):
#    def default(self,obj):
#        if obj.from_json
import unittest
import inspect
import numpy as np
import types

try:
    import qupulse_hdawg
except ImportError as err:
    raise unittest.SkipTest("qupulse_hdawg not present") from err
        
import tests.program.linspace_tests as linspace_tests
from qupulse_hdawg.seqc import HDAWGProgramManager
from qupulse.utils.types import TimeType


CT_SCHEMA_2405 = r"""{
  "$schema": "https://json-schema.org/draft-07/schema#",
  "title": "AWG Command Table Schema",
  "description": "Schema for ZI HDAWG AWG Command Table",
  "version": "1.2.1",
  "definitions": {
    "header": {
      "type": "object",
      "properties": {
        "version": {
          "type": "string",
          "pattern": "^1\\.[0-2](\\.[0-9]+)?$",
          "description": "File format version (Major.Minor / Major.Minor.Patch). This version must match with the relevant schema version."
        },
        "partial": {
          "description": "Set to true for incremental table updates",
          "type": "boolean",
          "default": false
        },
        "userString": {
          "description": "User-definable label",
          "type": "string",
          "maxLength": 30
        }
      },
      "required": [
        "version"
      ]
    },
    "table": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/entry"
      },
      "minItems": 0,
      "maxItems": 1024
    },
    "entry": {
      "type": "object",
      "properties": {
        "index": {
          "$ref": "#/definitions/tableindex"
        },
        "waveform": {
          "$ref": "#/definitions/waveform"
        },
        "phase0": {
          "$ref": "#/definitions/phase"
        },
        "phase1": {
          "$ref": "#/definitions/phase"
        },
        "amplitude0": {
          "$ref": "#/definitions/amplitude"
        },
        "amplitude1": {
          "$ref": "#/definitions/amplitude"
        }
      },
      "additionalProperties": false,
      "anyOf": [
        {
          "required": [
            "index",
            "waveform"
          ]
        },
        {
          "required": [
            "index",
            "phase0"
          ]
        },
        {
          "required": [
            "index",
            "phase1"
          ]
        },
        {
          "required": [
            "index",
            "amplitude0"
          ]
        },
        {
          "required": [
            "index",
            "amplitude1"
          ]
        }
      ]
    },
    "tableindex": {
      "type": "integer",
      "minimum": 0,
      "maximum": 1023
    },
    "waveform": {
      "type": "object",
      "properties": {
        "index": {
          "$ref": "#/definitions/waveformindex"
        },
        "length": {
          "$ref": "#/definitions/waveformlength"
        },
        "samplingRateDivider": {
          "$ref": "#/definitions/samplingratedivider"
        },
        "awgChannel0": {
          "$ref": "#/definitions/awgchannel"
        },
        "awgChannel1": {
          "$ref": "#/definitions/awgchannel"
        },
        "precompClear": {
          "$ref": "#/definitions/precompclear"
        },
        "playZero": {
          "$ref": "#/definitions/playzero"
        },
        "playHold": {
          "$ref": "#/definitions/playhold"
        }
      },
      "additionalProperties": false,
      "oneOf": [
        {
          "required": [
            "index"
          ]
        },
        {
          "required": [
            "playZero",
            "length"
          ]
        },
        {
          "required": [
            "playHold",
            "length"
          ]
        }
      ]
    },
    "waveformindex": {
      "description": "Index of the waveform to play as defined with the assignWaveIndex sequencer instruction",
      "type": "integer",
      "minimum": 0,
      "maximum": 15999
    },
    "waveformlength": {
      "description": "The length of the waveform in samples",
      "type": "integer",
      "multipleOf": 16,
      "minimum": 32
    },
    "samplingratedivider": {
      "descpription": "Integer exponent n of the sample rate divider: SampleRate / 2^n, n in range 0 ... 13",
      "type": "integer",
      "minimum": 0,
      "maximum": 13
    },
    "awgchannel": {
      "description": "Assign the given AWG channel to signal output 0 & 1",
      "type": "array",
      "minItems": 1,
      "maxItems": 2,
      "uniqueItems": true,
      "items": [
        {
          "type": "string",
          "enum": [
            "sigout0",
            "sigout1"
          ]
        }
      ]
    },
    "precompclear": {
      "description": "Set to true to clear the precompensation filters",
      "type": "boolean",
      "default": false
    },
    "playzero": {
      "description": "Play a zero-valued waveform for specified length of waveform, equivalent to the playZero sequencer instruction",
      "type": "boolean",
      "default": false
    },
    "playhold": {
      "description": "Hold the last played value for the specified number of samples, equivalent to the playHold sequencer instruction",
      "type": "boolean",
      "default": false
    },
    "phase": {
      "type": "object",
      "properties": {
        "value": {
          "description": "Phase value of the given sine generator in degree",
          "type": "number"
        },
        "increment": {
          "description": "Set to true for incremental phase value, or to false for absolute",
          "type": "boolean",
          "default": false
        }
      },
      "additionalProperties": false,
      "required": [
        "value"
      ]
    },
    "amplitude": {
      "type": "object",
      "properties": {
        "value": {
          "description": "Amplitude scaling factor of the given AWG channel",
          "type": "number",
          "minimum": -1.0,
          "maximum": 1.0
        },
        "increment": {
          "description": "Set to true for incremental amplitude value, or to false for absolute",
          "type": "boolean",
          "default": false
        },
        "register": {
          "description": "Index of amplitude register that is selected for scaling the pulse amplitude.",
          "type": "integer",
          "minimum": 0,
          "maximum": 3
        }
      },
      "additionalProperties": false,
      "anyOf": [
        {
          "required": [
            "value"
          ]
        },
        {
          "required": [
            "register"
          ]
        }
      ]
    }
  },
  "type": "object",
  "properties": {
    "$schema": {
      "type": "string"
    },
    "header": {
      "$ref": "#/definitions/header"
    },
    "table": {
      "$ref": "#/definitions/table"
    }
  },
  "additionalProperties": false,
  "required": [
    "header",
    "table"
  ]
}"""


class AllLinspaceTests(unittest.TestCase):
    
    def setUp(self):
        self.test_classes_list = [member for name, member in inspect.getmembers(linspace_tests, inspect.isclass)
                             if issubclass(member, unittest.TestCase) and not member.__name__=="TestCase"]

    def test_all(self):
        
        for test_class in self.test_classes_list:
            
            test_obj = test_class()
            test_obj.setUp()
            
            test_program, channels, markers = test_obj.return_program()
            
            awg = types.SimpleNamespace(num_channels=2, MAX_SAMPLE_RATE_DIVIDER=13, sample_rate_divider=0,
                                        master_device=types.SimpleNamespace(sample_clock=2.4)
                                        )
            
            test_manager = HDAWGProgramManager(awg, lambda idx: tuple([CT_SCHEMA_2405 for i in idx]))
            
            try:
                test_manager.add_program("test", test_program, channels, markers,
                                         np.ones(len(channels)),np.zeros(len(channels)),
                                         [None,]*len(channels), TimeType.from_fraction(24,10))
            except Exception as e:
                self.fail(f"{test_class.__name__} raised an exception: {e}")
                
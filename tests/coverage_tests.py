import qctoolkit.pulses.function_pulse_template
import qctoolkit.pulses.table_pulse_template

import tests.pulses.function_pulse_tests
import tests.pulses.table_pulse_template_tests

from tests.property_test_helper import assert_public_functions_tested_tester


FunctionPublicFunctionsTested = assert_public_functions_tested_tester(tests.pulses.function_pulse_tests,
                                                                      qctoolkit.pulses.function_pulse_template)
TablePublicFunctionsTested = assert_public_functions_tested_tester(tests.pulses.table_pulse_template_tests,
                                                                qctoolkit.pulses.table_pulse_template)
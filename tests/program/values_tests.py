import copy
import unittest
from unittest import TestCase

import numpy as np

from qupulse.pulses import *
from qupulse.program.linspace import *
from qupulse.program.transformation import *
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
from qupulse.program.values import DynamicLinearValue
from qupulse.utils.types import TimeType


class DynamicLinearValueTests(TestCase):
    def setUp(self):
        self.d = DynamicLinearValue(-100,{'a':np.pi,'b':np.e})
        self.d3 = DynamicLinearValue(-300,{'a':np.pi,'b':np.e})
    
    def test_value(self):
        dval = self.d.value({'a':12,'b':34})
        np.testing.assert_allclose(dval, 12*np.pi+34*np.e-100)
        
    # def test_abs(self):
    #     np.testing.assert_allclose(abs(self.d),100+np.pi+np.e)
    
    def test_add_sub_neg(self):

        self.assertEqual(self.d + 3,
                         DynamicLinearValue(-100+3,{'a':np.pi,'b':np.e}))
        self.assertEqual(self.d + np.pi,
                         DynamicLinearValue(-100+np.pi,{'a':np.pi,'b':np.e}))
        self.assertEqual(self.d + TimeType(12/5),
                         DynamicLinearValue(-100+TimeType(12/5),{'a':np.pi,'b':np.e}))
        #sub
        self.assertEqual(self.d - TimeType(12/5),
                         DynamicLinearValue(-100-TimeType(12/5),{'a':np.pi,'b':np.e}))
        
        #this would raise because of TimeType conversion
        # self.assertEqual(TimeType(12/5)-self.d,
        #                  DynamicLinearValue(100+TimeType(12/5),{'a':-np.pi,'b':-np.e}))
        #same type
        self.assertEqual(self.d+DynamicLinearValue(0.1,{'b':1,'c':2}),
                         DynamicLinearValue(-99.9,{'a':np.pi,'b':np.e+1,'c':2}))
    
    def test_mul(self):
        self.assertEqual(self.d*3,
                         DynamicLinearValue(-3*100,{'a':3*np.pi,'b':3*np.e}))
        self.assertEqual(3*self.d,
                         DynamicLinearValue(-3*100,{'a':3*np.pi,'b':3*np.e}))
        #div
        self.assertEqual(self.d3/3,
                         DynamicLinearValue(-100,{'a':np.pi/3,'b':np.e/3}))
        #raise
        self.assertRaises(TypeError,lambda: 3/self.d,)
    
    def test_eq(self):
        
        self.assertEqual(self.d==1,False)
        self.assertEqual(self.d==1+1j,False)
        # self.assertEqual(self.d>-101,True) #if one wants to allow these comparisons
        self.assertEqual(self.d<TimeType(24/5),True)
        
        self.assertEqual(self.d==self.d,True)
        self.assertEqual(self.d+1==self.d,False)
        self.assertEqual(self.d+1>self.d,False)
        self.assertEqual(self.d+1<self.d,False)
        self.assertEqual(self.d+1>=self.d,True)
        self.assertEqual(self.d+1<=self.d,False)
        
        self.assertEqual(self.d>self.d/2-51,True)
        self.assertEqual(self.d<self.d*2+101,True)
        
    def test_sympy(self):
        self.assertEqual(self.d._sympy_(), self.d)
        self.assertEqual(self.d.replace(1,1), self.d)
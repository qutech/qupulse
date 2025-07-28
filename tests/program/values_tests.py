import copy
import unittest
from unittest import TestCase

import numpy as np

from qupulse.pulses import *
from qupulse.program.linspace import *
from qupulse.program.transformation import *
from qupulse.pulses.function_pulse_template import FunctionPulseTemplate
from qupulse.program.values import DynamicLinearValue, ResolutionDependentValue
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
        # self.assertEqual(self.d<TimeType(24/5),True)  #if one wants to allow these comparisons
        
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
        
    def test_hash(self):
        self.assertEqual(hash(self.d), hash(self.d))
        self.assertNotEqual(hash(self.d), hash(self.d3))
        
        

class ResolutionDependentValueTests(TestCase):
    def setUp(self):
        self.d = ResolutionDependentValue((np.pi*1e-5,np.e*1e-5),(10,20),0.1)
        self.d2 = ResolutionDependentValue((np.e*1e-5,np.pi*1e-5),(10,20),-0.1)
        self.dtt = ResolutionDependentValue((TimeType(12,5),TimeType(14,5)),(10,20),TimeType(12,5))
        self.dint = ResolutionDependentValue((1,2),(10,20),1)
        
        self._default_res = 2**-16
        
    def test_call(self):
        val = self.d(self._default_res)
        #expected to round to 2*res each, so 60*2**16, offset also rounded
        expected_val = 2**-16 * 60 + 0.100006103515625
        self.assertAlmostEqual(val,expected_val,places=12)
        self.assertEqual((val/self._default_res)%1,0)
        
        #if no resolution must be tt or int
        self.assertEqual(self.dtt(None),TimeType(412, 5))
        self.assertEqual(self.dint(None),51)
        
    def test_dunder(self):
        self.assertEqual(bool(self.d), True)
        self.assertRaises(TypeError, lambda: self.d+self.d2)
        self.assertEqual(self.d-0.1,
                         ResolutionDependentValue((np.pi*1e-5,np.e*1e-5),(10,20),0.0))
        self.assertEqual(self.d*2,
                         ResolutionDependentValue((np.pi*1e-5*2,np.e*1e-5*2),(10,20),0.2))
        self.assertEqual(self.d/2,
                         ResolutionDependentValue((np.pi*1e-5/2,np.e*1e-5/2),(10,20),0.1/2)) 
        
        self.assertRaises(AssertionError, lambda: float(self.d))
        
        try: str(self.d)
        except: self.fail()
        
        self.assertEqual(hash(self.d), hash(self.d))
        self.assertNotEqual(hash(self.d), hash(self.d2))
        
        self.assertEqual(self.d==ResolutionDependentValue((np.pi*1e-5,np.e*1e-5),(10,20),0.1),True)
        
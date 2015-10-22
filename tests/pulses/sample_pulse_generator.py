import random

from qctoolkit.pulses.parameters import ConstantParameter
from qctoolkit.pulses.parameters import ParameterDeclaration
from qctoolkit.pulses.table_pulse_template import TablePulseTemplate

RANGE = 100
INTERPOLATION_STRATEGIES = list(TablePulseTemplate()._TablePulseTemplate__interpolation_strategies.keys())
#random.seed(2)

def getOrdered(length, min_=-RANGE, max_=RANGE,max_bound=None):
    a = [random.uniform(min_,max_) for _ in range(length)]
    a.sort()
    if max_bound and a[2] < max_bound:
        a[2] = random.uniform (max_bound,max_)
    return a
    
class SampleGenerator(object):
    def __init__(self, *args, **kwargs):
        object.__init__(self, *args, **kwargs)
        self.__ParameterDeclarationNameOffset = 0
        self.__TablePulseTemplateOffset = 0
        
    def generate_ConstantParameter(self, min_=-RANGE, max_=RANGE):
        while 1: yield ConstantParameter(random.uniform(min_,max_))
        
    def generate_ParameterDeclaration(self, min_=-RANGE, max_=RANGE, name_prefix="",max_bound=None):
        while 1: 
            bounds = getOrdered(3,min_,max_,max_bound)
            self.__ParameterDeclarationNameOffset+= 1
            yield ParameterDeclaration(name="ParameterDeclaration_{}".format(self.__ParameterDeclarationNameOffset),min=bounds[0],max=bounds[2],default=bounds[1])
            
    def generate_TablePulseTemplates(self,number_of_entries,max_dist=RANGE):
        while 1:
            x = TablePulseTemplate()
            name = "TablePulseTemplate_{}".format(self.__TablePulseTemplateOffset)
            self.__TablePulseTemplateOffset +=1
            if bool(random.getrandbits(1)):
                x.identifier = name
            previous_min = 0
            previous_max = 0
            parameter_names = []
            
            for i in range(number_of_entries):
                dict_ = {}
                for j in ["time","voltage"]:
                    a = random.choice(["float","str","ParameterDeclaration"])
                    if a == "float":
                        if j == "time":
                            dict_[j] = random.uniform(previous_max,previous_max+max_dist)
                            previous_min = dict_[j]
                            previous_max = dict_[j]
                        else:
                            dict_[j] = random.uniform(-RANGE,RANGE)
                    elif a == "str":
                        dict_[j] = "_".join([name,"str",str(i),str(j)])
                        parameter_names.append(dict_[j])
                    elif a == "ParameterDeclaration":
                        if j == "time":
                            dict_[j] = self.generate_ParameterDeclaration(previous_min, previous_min+max_dist, name, previous_max).__next__()
                            previous_min = dict_[j].min_value
                            previous_max = dict_[j].max_value
                        else:
                            dict_[j] = self.generate_ParameterDeclaration().__next__()
                        parameter_names.append(dict_[j].name)
                x.add_entry(time=dict_["time"],voltage=dict_["voltage"],interpolation=random.choice(INTERPOLATION_STRATEGIES))
            yield x
    
    def generate_SequencePulseTemplate(self):
        pass
            
    
class CounterExampleGenerator(object):
    def __init__(self, *args, **kwargs):
        object.__init__(self, *args, **kwargs)
        self.__ParameterDeclarationNameOffset = 0
        
    def generate_ConstantParameter(self):
        yield ConstantParameter(float('inf'))
        yield ConstantParameter(float('-inf'))
        
    def generate_ParameterDeclaration(self, max_=RANGE, min_=(-RANGE)):
        while 1: 
            bounds = [random.uniform(min_,max_) for _ in range(3)]
            while bounds == sorted(bounds):bounds = random.shuffle(bounds)
            self.__ParameterDeclarationNameOffset+= 1
            yield ParameterDeclaration(name="var_{}".format(self.__ParameterDeclarationNameOffset),min=bounds[0],max=bounds[2],default=bounds[1])
        
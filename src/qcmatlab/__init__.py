# intended use:
# from qctoolkit.qcmatlab import engine
# engine.workspace['max_variable'] = 42
# engine.workspace['foo'] = 'bar'
from .manager import Manager

engine = Manager.getEngine()
connect = Manager.connectTo
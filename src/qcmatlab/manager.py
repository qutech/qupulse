import matlab.engine
""" Start a MATLAB session or connect to a connect to an existing one. """

__all__ = ["Manager",
           "EngineNotFound",
           "NoConnectionSupported"]

class NoConnectionSupported(Exception):
    pass

class EngineNotFound(Exception):
    def __init__(self,searched_engine: str) -> None:
        self.searched_engine = searched_engine
    def __str__(self) -> str:
        return "Could not find the MATLAB engine with name {}".format(self.searched_engine)
        
class Manager():
    __engine_name = 'qc_toolkit_session'
    __engine_store = []
    
            
    @staticmethod        
    def __start_new_engine() -> None:
        Manager.__engine_store = matlab.engine.start_matlab('-desktop')
        
        if Manager.__engine_store.version('-release') == 'R2015a':
            print('Warning this ')
        else:
            Manager.__engine_store.matlab.engine.shareEngine(Manager.__engine_name,nargout=0)
        
    @staticmethod
    def __connect_to_existing_matlab_engine() -> None:
                
        # I found no nicer way to test the version of the python interface
        try:
            names = matlab.engine.find_matlab()
        except AttributeError:
            raise NoConnectionSupported()

        try:
            Manager.__engine_store = matlab.engine.connect_matlab(Manager.__engine_name)
        except matlab.engine.EngineError:
            raise EngineNotFound(names)
     
    @staticmethod
    def connectTo( engine_name: str ) -> None:
        
        if Manager.__engine_store is matlab.engine.matlabengine.MatlabEngine:
            if __engine_name == engine_name:
                print('Already connected to engine {name}.'.format(name=engine_name))
                return
        
        if engine_name in matlab.engine.find_matlab():
            if Manager.__engine_store is matlab.engine.matlabengine.MatlabEngine:
                print('Disconnecting from old engine...')
                Manager.__engine_store.quit()
            Manager.__engine_store = matlab.engine.connect_engine(engine_name)
            Manager.__engine_name = engine_name
        else:
            raise EngineNotFound(searched_engine=engine_name)
        
    @staticmethod
    def getEngine() -> matlab.engine.matlabengine.MatlabEngine:
        if Manager.__engine_store is matlab.engine.matlabengine.MatlabEngine:
            return Manager.__engine_store
        
        
        try:
            Manager.__connect_to_existing_matlab_engine()
        except NoConnectionSupported:
            print('Current MATLAB version does not support connection to running engines.\nCreating a new one...')
            Manager.__start_new_engine()
        except EngineNotFound:
            print('Could not find running engine with name {}.\nStarting a new one....'.format(Manager.__engine_name))
            Manager.__start_new_engine()
        
        assert( isinstance(Manager.__engine_store, matlab.engine.matlabengine.MatlabEngine) )
        return Manager.__engine_store
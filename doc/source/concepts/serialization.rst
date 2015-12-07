.. _serialization:

Serialization
-------------
Serialization and deserilization mechanisms were implemented to enable persistent storage and thus reusability of pulse template definitions. Currently, the serialization format is a plain text document containing JSON formatted data. [#format]_

Serialization was constructed in a way that allows that a given pulse template may refer to subtemplates which are used by several different parent templates (or more than once in one) such as, e.g., the measurement pulse. These should then be stored in a separate file and referenced by a unique identifier in all parent templates to avoid unnecessary redundancy. On the other hand, there might be subtemplates which are only relevant to their parent and thus should be embedded in its serialization to avoid creating a multitude of files that are meaningless to a user. To allow the serialization process to make this distinction, each pulse template (or other serializable object) provides an optional identifier (which can be set by the user via the constructor for all pulse template variants). If an identifier is present in a pulse template, it is stored in a separate file. If not, it is embedded in its parent's serialization.

The implementation of (de)serialization features :class:`.Serializer` class and a :class:`.Serializable` interface. Every class that implements the latter can be serialized and thus stored as a JSON file. Currently, this is the case for all :class:`.PulseTemplate` variants as well as the :class:`.ParameterDeclaration` class. Additionally, the :class:`.StorageBackend` interface abstracts the actual storage backend. While currently there only exists a single implementation of this interface, namely the :class:`.FileSystemStorageBackend`, this allows to support, e.g., database storage, in the future.

The essential methods of :class:`.Serializer` are :meth:`.Serializer.serialize` and :meth:`.Serializer.deserialize`. :meth:`.Serializer.serialize` serializes a serializable object (i.e., any object of a class that implements/derives from :class:`.Serializable`) in a recursive process: It invokes the :meth:`.Serializable.get_serialization_data` method of the provided serializable object, which in turn might invoke the :class:`.Sequencer` to obtain a serialization for complex embedded data (such as a :class:`.ParameterDeclaration` in a :class:`.TablePulseTemplate`). In the end, a dictionary representation of the object is obtained which is then converted into a JSON string using built-in Python functionality. The JSON representation is finally stored using a given :class:`.StorageBackend`. Deserialization (:meth:`.Serializer.deserialize`) works analogously and is thus not explained in detail here.

For an example of how to use serialization to store and load pulse templates, see .

.. note:: write examples

Implementing a :class:`.Serializable` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To make any new class serializable, it must derive from the :class:`.Serializable` and implement the methods :meth:`.Serializable.get_serialization_data`, :meth:`.Serializable.deserialize` and the :attr:`.Serializable.identifier` property.

If class objects should be stored in a separate file, the `identifier` must be a non-empty string. If, on the other hand, class objects should be embedded into their parent's serialization (as is the case for, e.g., :class:`.ParameterDeclaration`), `identifier` must be `None`.

The method `serialize` should return a dictionary of native Python types containing all relevant data. If the class has members that are not native Python types but must be serialized, they must be serializable and the `serialize` method can obtain their serialization as the return value of :meth:`.Serializer._serialize_subpulse` and embed it in its result. The dictionary returned by `serialize` should not include the identifier in the returned dictionary.

The method `deserialize` is invoked with all key-value pairs created by a call to `serialize` as keyword arguments as well as an additional `identifier` keyword argument (which may be `None`) and must return a valid corresponding class instance.

The following may serve as a simple example:
::
    class Foo(Serializable):
    
        def __init__(self, template: PulseTemplate, identifier: Optional[str]=None) -> None:
            self.__template = template
            self.__identifier = identifier
            
        @property
        def identifier(self) -> Optional[str]:
            return self.__identifier
            
        def get_serialization_data(self, serializer: Serializer) -> Dict[str, Any]:
            return dict(template=serializer._serialize_subpulse(self.__template))
            
        @staticmethod
        def deserialize(serializer: Serializer, template: Dict[str, Any], identifer: Optional[str]=None) -> Serializable:
            return Foo(serialzer.deserialize(template), identifier=identifier)
            

.. rubric:: Footnotes

.. [#format] After some discussion of the format in which to store the data, JSON files were the favored solution. The main competitor were relational SQL databases, which could provide a central, globally accessible pulse database. However, since pulses are often changed between experiments, a more flexible solution that can be maintained by users without database experience and also allows changes only in a local environment was desired. Storing pulse templates in files was the obvious solution to this. This greatest-simplicity-requirement was also imposed on the data format, which thus resulted in JSON being chosen over XML or other similar formats. An additional favorable argument for JSON is the fact that Python already provides methods that convert dictionaries containing only native python types into valid JSON and back.

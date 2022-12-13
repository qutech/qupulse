.. _serialization:

Pulse Storage and Serialization
-------------------------------
Serialization and deserialization mechanisms are implemented to enable persistent storage and thus reusability of pulse template definitions. Currently, the serialization format is a plain text document containing JSON formatted data. [#format]_

Serialization is constructed in a way that allows that a given pulse template may refer to subtemplates which are used by several different parent templates (or more than once in one) such as, e.g., a measurement pulse.

These subtemplates are stored in a separate file and referenced by a unique identifier in all parent templates. On the other hand, there might be subtemplates which are only relevant to their parent and thus should be embedded in its serialization to avoid creating a multitude of files that have no value to the user. To allow the serialization process to make this distinction, each pulse template (or other serializable object) provides an optional identifier (which can be set by the user via the constructor for all pulse template variants). If an identifier is present in a pulse template, it is stored in a separate file. If not, it is embedded in its parent's serialization.

The implementation of (de)serialization mainly relies on the :class:`.PulseStorage` class and the :class:`.Serializable` interface. Every class that implements the latter can be serialized and thus stored as a JSON file. Currently, this is the case for all :class:`.PulseTemplate` variants as well as the :class:`.ParameterConstraint` class.

The :class:`.PulseStorage` offers a convenient dictionary-like interface for storing and retrieving pulse template objects (or other objects of type :class:`.Serializable`) to the user. It is responsible for transparently invoking the actual JSON encoding (or decoding) of :class:`.Serializable` objects including dealing with handling references to subtemplates with identifiers.

Finally, the :class:`.StorageBackend` interface abstracts the actual storage backend. While currently there only exists a few implementations of this interface, most importantly the :class:`.FilesystemStorageBackend`, this allows to support, e.g., database storage, in the future. :class:`.PulseStorage` requires an instance of :class:`.StorageBackend` which represents its persistent pulse storage during initialization.

For an example of how to use :class:`.PulseStorage` to store and load pulse templates, see :ref:`/examples/01PulseStorage.ipynb` in the examples section.

Global Pulse Registry
^^^^^^^^^^^^^^^^^^^^^^

qupulse features the concept of a pulse registry, i.e., a global dictionary-like object that keeps track of all named
pulse templates (and other :class:`.Serializable` instances with identifiers) in the program to ensure that no identifier
is used twice accidentally. Every :class:`.Serializable` instance automatically registers in the registry during object
construction and will raise an error if the given identifier is already taken.

To manage separate registries, every :class:`.Serializable` (sub)class has an optional construction argument ``registry``
to indicate the registry to use, although this should be used rarely as the presence of several distinct registries
undermines the intended purpose of preventing duplicated identifiers for serializable objects. If the ``registry``
argument is not specified, the default global pulse registry is used.

:class:`.PulseStorage` can (and should) be used as the pulse registry. Use the :meth:`.PulseStorage.set_to_default_registry`
method to set any :class:`.PulseStorage` object as the central registry.


Implementing a :class:`.Serializable` Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To make any new class serializable, it must derive from the :class:`.Serializable` and implement the methods :meth:`.Serializable.get_serialization_data`, :meth:`.Serializable.deserialize` and the :attr:`.Serializable.identifier` property.

If class objects should be stored in a separate file, the `identifier` must be a non-empty string. If, on the other hand, class objects should be embedded into their parent's serialization (as is the case for, e.g., :class:`.ParameterConstraint`), :attr:`.Serializable.identifier` must be `None`.

The :class:`.Serializable` class takes care of handling the identifier. Deriving classes must forward the ``identifier``
argument in the ``__init__`` method to :class:`.Serializable.__init__()`. Additionally, to comply with the pulse registry,
deriving classes must call :class:`.Serializable._register` at the end of their own ``__init__`` method, *after* the
object is completely assembled (and can potentially be serialized).

The method :meth:`.Serializable.get_serialization_data` should return a dictionary of containing all relevant data. The objects contained
in the returned dictionary can be of any native Python type, sets, lists or dictionary as well as of type
:class:`.Serializable`. Note that nested :class:`.Serializable` objects, e.g., subtemplates of a pulse template,
should be contained as is in the dictionary returned, i.e., ``get_serialization_data`` should
not make recursive calls to ``get_serialization_data`` of nested objects.
The :class:`.Serializable` class provides an implementation for :meth:`.Serializable.get_serialization_data` which returns
a dictionary containing information about type and identifier. This should be called at the beginning of implementations
of :meth:`.Serializable.get_serialization_data` in any derived class and all further information added to the dictionary
thus obtained.

The method :meth:`.Serializable.deserialize` is invoked with all key-value pairs created by a call to  :meth:`.Serializable.get_serialization_data` as keyword arguments
as well as an additional ``identifier`` keyword argument (which may be ``None``) and must return a valid corresponding
class instance. :class:`.Serializable` provides a default implementation which forwards all incoming keyword
arguments to the classes ``__init__`` method, which is sufficient in most cases. Derived classes only need to implement
``deserialize`` if they need to tweak the incoming keyword arguments before construction the corresponding class instance.
An example for this is :class:`.SequencePulseTemplate`.

The following code snippet may serve as an example for a simple implementation of a serializable class:

.. code-block:: python

    from qupulse.serialization import Serializable, PulseRegistryType
    from typing import Any, Dict, Optional

    class Foo(Serializable):
        def __init__(self,
                     template: Serializable,
                     mapping: Dict[str, int],
                     identifier: Optional[str]=None, registry:
                     PulseRegistryType=None) -> None:

            super().__init__(identifier=identifier)
            self.__template = template
            self.__mapping = mapping
            self._register(registry)

        def get_serialization_data(self) -> Dict[str, Any]:
            data = super().get_serialization_data()
            data['template'] = self.__template
            data['mapping'] = self.__mapping
            return data

.. rubric:: Footnotes

.. [#format] After some discussion of the format in which to store the data, JSON files were the favored solution. The main competitor were relational SQL databases, which could provide a central, globally accessible pulse database. However, since pulses are often changed between experiments, a more flexible solution that can be maintained by users without database experience and also allows changes only in a local environment was desired. Storing pulse templates in files was the obvious solution to this. This greatest-simplicity-requirement was also imposed on the data format, which thus resulted in JSON being chosen over XML or other similar formats. An additional favorable argument for JSON is the fact that Python already provides methods that convert dictionaries containing only native python types into valid JSON and back.
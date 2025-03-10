"""
Utility classes and methods to pickle parts of symbolic graph.

These pickled graphs can be used, for instance, as cases for
unit tests or regression tests.
"""

import os
import pickle
import sys
import tempfile
import zipfile
from collections import defaultdict
from contextlib import closing
from io import BytesIO
from pickle import HIGHEST_PROTOCOL

import numpy as np

import pytensor


try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL

from pytensor.compile.sharedvalue import SharedVariable


__docformat__ = "restructuredtext en"
__authors__ = "Pascal Lamblin " "PyMC Developers " "PyTensor Developers "
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"


min_recursion = 3000
if sys.getrecursionlimit() < min_recursion:
    sys.setrecursionlimit(min_recursion)

Pickler = pickle.Pickler


class StripPickler(Pickler):
    """Subclass of `Pickler` that strips unnecessary attributes from PyTensor objects.

    Example
    -------

        fn_args = dict(inputs=inputs,
                       outputs=outputs,
                       updates=updates)
        dest_pkl = 'my_test.pkl'
        with open(dest_pkl, 'wb') as f:
            strip_pickler = StripPickler(f, protocol=-1)
            strip_pickler.dump(fn_args)

    """

    def __init__(self, file, protocol=0, extra_tag_to_remove=None):
        # Can't use super as Pickler isn't a new style class
        super().__init__(file, protocol)
        self.tag_to_remove = ["trace", "test_value"]
        if extra_tag_to_remove:
            self.tag_to_remove.extend(extra_tag_to_remove)

    def save(self, obj):
        # Remove the tag.trace attribute from Variable and Apply nodes
        if isinstance(obj, pytensor.graph.utils.Scratchpad):
            for tag in self.tag_to_remove:
                if hasattr(obj, tag):
                    del obj.__dict__[tag]
        # Remove manually-added docstring of Elemwise ops
        elif isinstance(obj, pytensor.tensor.elemwise.Elemwise):
            if "__doc__" in obj.__dict__:
                del obj.__dict__["__doc__"]

        return Pickler.save(self, obj)


class PersistentNdarrayID:
    """Persist ndarrays in an object by saving them to a zip file.

    :param zip_file: A zip file handle that the NumPy arrays will be saved to.
    :type zip_file: :class:`zipfile.ZipFile`


    .. note:
        The convention for persistent ids given by this class and its derived
        classes is that the name should take the form `type.name` where `type`
        can be used by the persistent loader to determine how to load the
        object, while `name` is human-readable and as descriptive as possible.

    """

    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.count = 0
        self.seen = {}

    def _resolve_name(self, obj):
        """Determine the name the object should be saved under."""
        name = f"array_{self.count}"
        self.count += 1
        return name

    def __call__(self, obj):
        if isinstance(obj, np.ndarray):
            if id(obj) not in self.seen:

                def write_array(f):
                    np.lib.format.write_array(f, obj)

                name = self._resolve_name(obj)
                zipadd(write_array, self.zip_file, name)
                self.seen[id(obj)] = f"ndarray.{name}"
            return self.seen[id(obj)]


class PersistentSharedVariableID(PersistentNdarrayID):
    """Uses shared variable names when persisting to zip file.

    If a shared variable has a name, this name is used as the name of the
    NPY file inside of the zip file. NumPy arrays that aren't matched to a
    shared variable are persisted as usual (i.e. `array_0`, `array_1`,
    etc.)

    :param allow_unnamed: Allow shared variables without a name to be
        persisted. Defaults to ``True``.
    :type allow_unnamed: bool, optional

    :param allow_duplicates: Allow multiple shared variables to have the same
        name, in which case they will be numbered e.g. `x`, `x_2`, `x_3`, etc.
        Defaults to ``True``.
    :type allow_duplicates: bool, optional

    :raises ValueError
        If an unnamed shared variable is encountered and `allow_unnamed` is
        ``False``, or if two shared variables have the same name, and
        `allow_duplicates` is ``False``.

    """

    def __init__(self, zip_file, allow_unnamed=True, allow_duplicates=True):
        super().__init__(zip_file)
        self.name_counter = defaultdict(int)
        self.ndarray_names = {}
        self.allow_unnamed = allow_unnamed
        self.allow_duplicates = allow_duplicates

    def _resolve_name(self, obj):
        if id(obj) in self.ndarray_names:
            name = self.ndarray_names[id(obj)]
            count = self.name_counter[name]
            self.name_counter[name] += 1
            if count:
                if not self.allow_duplicates:
                    raise ValueError(
                        f"multiple shared variables with the name `{name}` found"
                    )
                name = f"{name}_{count + 1}"
            return name
        return super()._resolve_name(obj)

    def __call__(self, obj):
        if isinstance(obj, SharedVariable):
            if obj.name:
                if obj.name == "pkl":
                    ValueError("can't pickle shared variable with name `pkl`")
                self.ndarray_names[id(obj.container.storage[0])] = obj.name
            elif not self.allow_unnamed:
                raise ValueError(f"unnamed shared variable, {obj}")
        return super().__call__(obj)


class PersistentNdarrayLoad:
    """Load NumPy arrays that were persisted to a zip file when pickling.

    :param zip_file: The zip file handle in which the NumPy arrays are saved.
    :type zip_file: :class:`zipfile.ZipFile`

    """

    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.cache = {}

    def __call__(self, persid):
        array_type, name = persid.split(".")
        del array_type
        # array_type was used for switching gpu/cpu arrays
        # it is better to put these into sublclasses properly
        # this is more work but better logic
        if name in self.cache:
            return self.cache[name]
        ret = None
        with self.zip_file.open(name) as f:
            ret = np.lib.format.read_array(f)
        self.cache[name] = ret
        return ret


def dump(
    obj,
    file_handler,
    protocol=DEFAULT_PROTOCOL,
    persistent_id=PersistentSharedVariableID,
):
    """Pickles an object to a zip file using external persistence.

    :param obj: The object to pickle.
    :type obj: object

    :param file_handler: The file handle to save the object to.
    :type file_handler: file

    :param protocol: The pickling protocol to use. Unlike Python's built-in
        pickle, the default is set to `2` instead of 0 for Python 2. The
        Python 3 default (level 3) is maintained.
    :type protocol: int, optional

    :param persistent_id: The callable that persists certain objects in the
        object hierarchy to separate files inside of the zip file. For example,
        :class:`PersistentNdarrayID` saves any :class:`numpy.ndarray` to a
        separate NPY file inside of the zip file.
    :type persistent_id: callable

    .. versionadded:: 0.8

    .. note::
        The final file is simply a zipped file containing at least one file,
        `pkl`, which contains the pickled object. It can contain any other
        number of external objects. Note that the zip files are compatible with
        NumPy's :func:`numpy.load` function.

    >>> import pytensor
    >>> foo_1 = pytensor.shared(0, name='foo')
    >>> foo_2 = pytensor.shared(1, name='foo')
    >>> with open('model.zip', 'wb') as f:
    ...     dump((foo_1, foo_2, np.array(2)), f)
    >>> np.load('model.zip').keys()
    ['foo', 'foo_2', 'array_0', 'pkl']
    >>> np.load('model.zip')['foo']
    array(0)
    >>> with open('model.zip', 'rb') as f:
    ...     foo_1, foo_2, array = load(f)
    >>> array
    array(2)

    """
    with closing(
        zipfile.ZipFile(file_handler, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
    ) as zip_file:

        def func(f):
            p = pickle.Pickler(f, protocol=protocol)
            p.persistent_id = persistent_id(zip_file)
            p.dump(obj)

        zipadd(func, zip_file, "pkl")


def load(f, persistent_load=PersistentNdarrayLoad):
    """Load a file that was dumped to a zip file.

    :param f: The file handle to the zip file to load the object from.
    :type f: file

    :param persistent_load: The persistent loading function to use for
        unpickling. This must be compatible with the `persisten_id` function
        used when pickling.
    :type persistent_load: callable, optional

    .. versionadded:: 0.8
    """
    with closing(zipfile.ZipFile(f, "r")) as zip_file:
        p = pickle.Unpickler(BytesIO(zip_file.open("pkl").read()))
        p.persistent_load = persistent_load(zip_file)
        return p.load()


def zipadd(func, zip_file, name):
    """Calls a function with a file object, saving it to a zip file.

    :param func: The function to call.
    :type func: callable

    :param zip_file: The zip file that `func` should write its data to.
    :type zip_file: :class:`zipfile.ZipFile`

    :param name: The name of the file inside of the zipped archive that `func`
        should save its data to.
    :type name: str

    """
    with tempfile.NamedTemporaryFile("wb", delete=False) as temp_file:
        func(temp_file)
        temp_file.close()
        zip_file.write(temp_file.name, arcname=name)
    if os.path.isfile(temp_file.name):
        os.remove(temp_file.name)

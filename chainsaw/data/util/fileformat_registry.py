
class FileFormatUnsupported(Exception):
    """ No available reader is able to handle the given extension."""


class FileFormatRegistry(object):
    """ Registry for trajectory file objects. """
    _readers = {}

    @classmethod
    def register(cls, *args):
        """ register the given class as Reader class for given extension(s). """
        def decorator(f):
            if hasattr(f, 'SUPPORTED_EXTENSIONS') and f.SUPPORTED_EXTENSIONS is not ():
                raise RuntimeError("please call register() only once per class!")
            extensions = tuple(args)
            cls._readers.update({e: f for e in extensions})
            f.SUPPORTED_EXTENSIONS = extensions
            return f
        return decorator

    @staticmethod
    def is_md_format(extension):
        try:
            from pyemma.coordinates.data import FeatureReader
            return extension in FeatureReader.SUPPORTED_EXTENSIONS
        except ImportError:
            # TODO: this could cause problems
            return False

    def supported_extensions(self):
        return self._readers.keys()

    def __getitem__(self, item):
        try:
            return self._readers[item]
        except KeyError:
            raise FileFormatUnsupported("Extension {ext} is not supported by any reader.".format(ext=item))

# singleton pattern
FileFormatRegistry = FileFormatRegistry()

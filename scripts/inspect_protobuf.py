import google.protobuf as pb
import inspect
print('pb version:', getattr(pb, '__version__', 'unknown'))
print('pb file:', pb.__file__)
import pkgutil
print('google package contents sample:', pkgutil.iter_modules([pb.__path__[0]]).__class__)

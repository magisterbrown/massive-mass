from setuptools import setup, find_packages
#import re
#
#try:
#    from pip._internal.operations import freeze
#except ImportError:
#    from pip.operations import freeze
#
#installed_names = map(lambda x:re.match('\w+',x).group(), freeze.freeze())
#
#exceptions = [
#        '@ file',
#        'nvidia',
#    ]
#
#with open('requirements.txt', 'r') as f:
#    deps = f.readlines()
#    
#    #Remove already installed
#    for insn in installed_names:
#        deps = list(filter(lambda x:insn != re.match('\w+',x).group(), deps))
#
#    #Remove packages that have patterns from the exceptios
#    for exc in exceptions:
#        deps = list(filter(lambda x:not (exc in x), deps))
#
#    print(deps)
#

setup(name='biopack',
      version='1.0',
      description='',
      author='magisterbrownie',
      author_email='magisterbrownie@gmail.com',
      url='',
      packages=find_packages(),
     )


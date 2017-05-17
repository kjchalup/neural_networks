"""setup.py for the neural_networks package."""
import os
from setuptools import setup

def read(fname):
    """Utiltiy function for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='neural_networks',
      version='0.1.0',
      description='Neural network modules in Tensorflow.',
      # url='',
      author='Krzysztof Chalupka',
      author_email='janchatko@gmail.com',
      license='GPL 3.0',
      packages=['neural_networks'],
      install_requires=[
          'sklearn==0.18.1',
          'numpy==1.12.1',
          'tensorflow==1.0.0',
      ],
      long_description=read('README.rst'),
      keywords=['neural networks', 'machine learning'],
     )

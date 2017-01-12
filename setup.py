from setuptools import setup
from setuptools import find_packages

setup(name='ica',
      version='1.0',
      description='Python implementation of the Iterative Classification Algorithm',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='https://github.com/tkipf/ica',
      license='MIT',
      install_requires=['numpy',
                        'networkx',
                        'sklearn',
                        'scipy'
                        ],
      package_data={'ica': ['README.md', 'ica/data/']},
      packages=find_packages())
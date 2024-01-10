from setuptools import setup
from setuptools import find_packages

setup(name='cnn_gcn',
      version='1.0',
      description='CNN Implementation Tensorflow',
      author='zulkarnainprastyo',
      author_email='zulkarnain.prastyo@binus.ac.id',
      download_url='https://github.com/zulkarnainprastyo/cnn_gcn',
      license='MIT',
      install_requires=['numpy>=1.15.4',
                        'tensorflow>=1.15.2,<2.0',
                        'networkx>=2.2',
                        'scipy>=1.1.0'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())
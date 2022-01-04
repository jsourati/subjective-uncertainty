#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os, sys

with open('README.md') as readme_file:
    readme = readme_file.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

data_dir = os.path.join(sys.prefix, "local/lib/python2.7/dist-packages/biocertainty")

    
setup(name='biocertainty',	
    version='1.0.8',	
    description='Python package that provides the certainty about biomedical statements.',	
    url='https://github.com/Guindillator/BioCertainty',	
    author='Mario Prieto',	
    author_email='mario.prieto@upm.es',	
    license='Wilkinson Laboratory',
    packages=find_packages(),
    install_requires=requirements,
    long_description=readme,
    package_dir={'biocertainty':
                 'biocertainty'},	
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7'
    ],
    test_suite='tests',
#     install_requires=requirements,
#     data_files=[('', ['data/training_set.csv'])],
#     package_data = {'': ['data/training_set.csv']},
    data_files   = [(os.path.join(data_dir),  [("data/training_set.csv"),
                                     ("data/model.json"), 
                                    ("data/model.h5")])],
#     package_data = {"biocertainty":  [( "data/training_set.csv"),
#                                      ( "data/model.json"), 
#                                      ( "data/model.h5")]},
# #     package_data = {"biocertainty":  [os.path.join(data_dir, "data/training_set.csv"),
# #                                      os.path.join(data_dir, "data/model.json"), 
# #                                      os.path.join(data_dir, "data/model.h5")]},
    include_package_data = True, 
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ]
)


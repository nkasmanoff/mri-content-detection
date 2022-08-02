#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='src',
      version='0.0.1',
      description='MRI Content Detection',
      author='Noah Kasmanoff',
      author_email='nsk367@nyu.edu',
      url='https://github.com/nkasmanoff/mri_content_detection',
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name='ensemble_boxes',
    version='0.0dev',
    author='Alexander Soare',
    packages=['cv_utils'],
    url='https://github.com/alexander-soare/CV-Utils',
    license='Apache 2.0',
    description='My own miscellaneous helpers for computer vision developement',
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-python',
    ],
)
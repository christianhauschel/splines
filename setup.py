from setuptools import setup, find_packages

setup(
    name='splines',
    version='0.1',
    author="Christian Hauschel",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        "geomdl",
        "scikit-learn",
        "matplotlib",
    ],
)

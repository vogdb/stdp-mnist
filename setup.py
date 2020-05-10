from setuptools import setup, find_packages

with open('README.txt') as f:
    README = f.read()

setup(
    name='stdp-mnist',
    install_requires=[
        'brian2',
        'git+https://github.com/datapythonista/mnist',
    ],
    packages=find_packages(),
    version='0.0.1',
    author='',
    author_email='',
    description='',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/vogdb/stdp-mnist',
    classifiers=[]
)


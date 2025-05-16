from setuptools import setup, find_packages

setup(
    name='fm4g',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.
        # 'numpy>=1.18.0',
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here, e.g.
            # 'gridfm=gridfm.cli:main',
        ],
    },
    author='FM4G',
    author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AI4S-PPFL/GridFM',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

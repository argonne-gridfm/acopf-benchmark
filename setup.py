from setuptools import setup, find_packages
import subprocess

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
    author_email='gridfm@anl.gov',
    description='GridFM: P',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/argonne-gridfm/acopf-benchmark',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)


def check_nvcc_version(min_version='12'):
    try:
        output = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
        for line in output.split('\n'):
            if 'release' in line:
                version_str = line.split('release')[-1].split(',')[0].strip()
                major_version = int(version_str.split('.')[0])
                if major_version < int(min_version):
                    raise RuntimeError(f'nvcc version >= {min_version} is required, found {version_str}')
                break
        else:
            raise RuntimeError('Could not determine nvcc version.')
    except FileNotFoundError:
        raise RuntimeError('nvcc not found. Please install CUDA Toolkit >= 12.')
    except Exception as e:
        raise RuntimeError(f'Error checking nvcc version: {e}')


check_nvcc_version('12')

from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='siamics',
    version='1.0',
    packages=['siamics', 'siamics.data', 'siamics.models', 'siamics.models.scGPT', 'siamics.utils'],

    description='SiaMics: Genomics deep learning library',
    author='Behnam (Ben) Maneshgar',
    author_email='maneshgar.behnam@gmail.com',
    url='https://github.com/bmaneshgar/SiaMics',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=['SiaMics', 'Genomics', 'Deep Learning', 'BioTech'],
    classifiers=[],

    py_modules=['cli'],
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
        ]
    }
)

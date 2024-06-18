from setuptools import setup, find_packages

setup(
    name="mandr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'Flask', 'srsly', 'scikit-learn', 'pandas', 'altair', 'diskcache', 'rich', 'markdown'
    ],
)

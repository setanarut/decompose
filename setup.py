from setuptools import setup, find_packages


setup(
    name="decompose",
    author="setanarut",
    version="0.1.0",
    url="https://github.com/setanarut/decompose",
    packages=find_packages(),
    description="Decompose image into layers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.26.1",
        "Pillow==10.1.0",
        "guided_filter_pytorch==3.7.5",
        "torch==2.1.0",
    ],
    python_requires=">=3.11",
)

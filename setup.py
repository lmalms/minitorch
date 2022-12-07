from setuptools import find_packages, setup

setup(
    name="minitorch",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "hypothesis>=6.54.6",
        "matplotlib==3.5.1",
        "numba>=0.56",
        "numpy>=1.23.3",
        "pytest==7.2.0",
        "streamlit>=1.11.1",
    ],
)

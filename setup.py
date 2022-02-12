from setuptools import setup, find_packages

setup(
    name="minitorch",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "hypothesis==4.38.0",
        "pytest==6.0.1"
    ]
)
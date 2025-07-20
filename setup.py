# setup.py
from setuptools import setup, find_packages

setup(
    name="RIVET",
    version="0.1.0",
    description="A quick-test XPBD physics engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # list run-time dependencies here, e.g. "numpy", "pygame"
    ],
)
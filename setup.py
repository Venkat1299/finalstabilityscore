from setuptools import setup, find_packages

setup(
    name="stabilityscore",
    version="1.0.0",
    author="Venkata Reddy",
    description="Robust Stability Score framework for evaluating dataset sensitivity under perturbations",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "joblib"
    ],
    python_requires=">=3.7",
)

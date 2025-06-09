from setuptools import setup, find_packages

setup(
    name="car-price-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "scikit-learn",
        "joblib",
        "python-multipart",
    ],
    python_requires=">=3.10",
) 
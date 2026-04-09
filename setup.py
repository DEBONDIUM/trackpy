from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="trackpy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
    ],
    author="Luc Brémaud, Jérémie Girardot, Vincent Fournier",
    description="Simple object tracking in image sequences using OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DEBONDIUM/trackpy",  # à modifier
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="e2ml",
    version="0.0.0",
    author="Klara Gutekunst",
    author_email="klara.gutekunst@student.uni-kassel.de",
    description="This package contains methods implemented during the course"
                "'Experimentation and Evaluation in Machine Learning' (E2ML)"
                "of the 'Intelligent Embedded Systems' (IES) department at "
                "the University of Kassel in Germany.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy~=1.25.1',
        'scipy~=1.11.1',
        'seaborn~=0.12.2',
        'scikit-learn~=1.2.0',
        'matplotlib~=3.3.4',
        'iteration-utilities~=0.11.0',
        'jupyter~=1.0.0',
        'pandas~=1.3.3'
    ],
)

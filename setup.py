from setuptools import find_packages, setup

setup(
    name="anypick_dk",
    version="0.0.0",
    author="Karen Guo, Toya Takahashi",
    author_email="karguo@mit.edu, toyat@mit.edu",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml"]},
)
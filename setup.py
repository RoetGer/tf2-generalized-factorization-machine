import os.path as osp
from setuptools import setup, find_packages


# read the contents of your README file
rep_folder = osp.abspath(osp.dirname(__file__))
with open(osp.join(rep_folder, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tf2_gfm",
    version="0.0.1",
    url="https://github.com/RoetGer/tf2-generalized-factorization-machine",
    packages=find_packages(include=["tf2_gfm"]),
    long_description=long_description, 
    long_description_content_type="text/markdown"
)


# CMS-ML-analysis
A repository to aggregate some tools for HEP analysis, using ML.

# 


## Besides the usual python packages, to do an modern analysis, we will need some additional tools.
The most important ones are:

- awkward
- uproot
- fsspec-xrootd
- vector
- coffea

TODO:
- [ ] Create a first principles notebook from OpenData School


# Requirements

- python3.10
- Ubuntu 24.04

$sudo apt-get install cmake
$sudo apt-get install uuid-dev
$sudo apt-get install libssl-dev
$sudo apt-get install xrootd

# Create a virtual enviroment

python -m venv .venv # you can also choose another name, like .cmsml, or .myvenv

## After create the .venv 
- This could take a while

pip install -r requirements.txt
!pip install --upgrade pip
!pip install --upgrade awkward
!pip install --upgrade uproot
!pip install --upgrade pandas
!pip install --upgrade coffea
!pip install --upgrade matplotlib

# Follow the notebooks


# Create a virtual enviroment


# TODO:
- [ ] Configure the model architecture
- [ ] Get better results
- [ ] Choose and work with a MET dataset
- [ ] Mono photon signal detection
- [ ] Configure and make functions plots.

# CMS-ML-analysis
A repository to aggregate some tools for HEP analysis using NanoAOD format, also using ML.
We are following some of the contents from the last CMS Open Data School [https://cms-opendata-workshop.github.io/2024-07-29-CERN/]



## Besides the usual python packages, to do a modern analysis, we will need some additional tools.
The most important ones are:

- awkward
- uproot
- xrootd
- vector
- coffea


# Requirements
For these analysis, I used:

- working PC
- internet
- Ubuntu 24.04
- python3.10


# Preparation


## Install packages

First, if you are not using a docker, install those:

```$ sudo apt-get install cmake  uuid-dev  libssl-dev xrootd-client xrootd-server python3-xrootd```

## Create a virtual enviroment

```python -m venv .venv  ```

You can also choose another name, like ```.cmsml```, or ```.myvenv```

### After the .venv creation 

This could take a while

```$ pip install -r requirements.txt ```
```$ pip install --upgrade pip```
```$ pip install --upgrade awkward```
```$ pip install --upgrade uproot```
```$ pip install --upgrade pandas```
```$ pip install --upgrade coffea```
```$ pip install --upgrade matplotlib```

# Next step: Follow the notebooks

In the ```notebooks``` folder. 
For this, I recommend using VSCode with Jupyter extensions (my case) or Jupyterlab itself.



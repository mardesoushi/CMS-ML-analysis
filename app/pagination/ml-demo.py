import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

###### From the notebook...
import logging
import time
import awkward as ak
import cabinetry
import cloudpickle
import correctionlib
# from coffea import processor
# from coffea.nanoevents import NanoAODSchema
# from coffea.analysis_tools import PackedSelection
import copy
import hist
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import requests
#import utils  # contains code for bookkeeping and cosmetics, as well as some boilerplate
logging.getLogger("cabinetry").setLevel(logging.INFO)
import sys
sys.path.append('../')
from modules.prepare_data import *
##
# The classics
import numpy as np
import matplotlib.pylab as plt
import matplotlib # To get the version
import pandas as pd

# The newcomers
import awkward as ak
import uproot
import vector
vector.register_awkward()
import requests
import os
import time
import json
import subprocess


st.write("Under construction...")
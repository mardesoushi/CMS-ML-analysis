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
st.set_page_config(page_title="CMS Open Data", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")


def get_page():
    'Seleciona a pagina de acordo com a pagina atual, se a pagina selecionada for diferente, muda de pagina'

    pages_list = []
    start = st.Page("pagination/home.py", title="Home", icon=':material/home:')
    data_selection_pg = st.Page("pagination/data-selection.py", title="Data Selection", icon=':material/account_tree:')
    ml_demo_pg = st.Page("pagination/ml-demo.py", title="ML Demo", icon=':material/psychology:')

    
    ## Acesso do paciente:manufacturing
    pages_list.append(start)
    pages_list.append(data_selection_pg)
    pages_list.append(ml_demo_pg)

    pg = st.navigation(pages_list)

    pg.run()

    return pg
    

def get_logo_sidebar():
    '''Insere a logo na barra lateral com um tamanho maior'''

    markdown_logo = """<style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        height: 5rem;
        width: auto;
    }
    
    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
        display: flex;
        align-items: center;
    }
    </style>"""
    st.markdown(markdown_logo, unsafe_allow_html =True)
    st.logo('resources/sprace-logo.png')

get_logo_sidebar()
get_page()




## Export this libpath to streamlit sees any changes in the folders
#export PYTHONPATH=$PYTHONPATH:/mnt/mnt/d/codigos/posdoc/CMS-ML-analysis/pagination/
#export PYTHONPATH=$PYTHONPATH:/path/to/folder/CMS-ML-analysis/
#export PYTHONPATH=$PYTHONPATH:/path/to/folder/CMS-ML-analysis/modules/
#export PYTHONPATH=$PYTHONPATH:/path/to/folder/CMS-ML-analysis/app/

# Read the file and split it into a list of root link strings

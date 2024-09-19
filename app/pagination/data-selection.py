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


# TODO: 
    # Apply Trigger selection, filter parameters by name
    # Create different dataframes


file_example = 'https://opendata.cern.ch/record/30526/files/CMS_Run2016G_MET_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_270000_file_index.txt'
st.session_state.file_link = st.text_input("""Link to ```.txt``` file""", value=file_example, placeholder=file_example)
st.info(f'For example: \n\n ```{file_example}```')
def download_data_file():
    ## Get a link to download .root files in .txt. format

    ## Download the file
    downloaded_file = 'downloaded_file.txt'
    subprocess.run(["wget", "-O", downloaded_file,  st.session_state.file_link ])

if st.button('Download data file'):
    download_data_file()

downloaded_file = 'downloaded_file.txt'
with open(downloaded_file, 'r', encoding='UTF-8') as file:
    root_files_list = file.read().splitlines()


## Show downloaded files list
with st.expander('Root files list'):
    for file in root_files_list:
        st.text(file)
#st.write(root_files_list)



## Openfirst file to get metadata
if 'file_raw' not in st.session_state:
    st.write("Open first file from the list to get metadata:")
    with st.spinner(f'Opening file with uproot... File: {root_files_list[0]}'):
        try:
            file_raw = uproot.open(root_files_list[0])
            st.session_state.file_raw = file_raw
        except:
            print(f"Could not open {root_files_list[0]}")
            st.error(f"Could not open {root_files_list[0]}")

events = st.session_state.file_raw['Events']
nevents = events.num_entries
st.write(f'Number of events in the file: {nevents}')

raw_parameter = st.selectbox('File raw parameters', st.session_state.file_raw.keys(), index=1 )

if raw_parameter == 'Events;1':
    sub_parameter_list = st.multiselect(f'Parameters for {raw_parameter}:',default=['MET_pt'], options=st.session_state.file_raw[raw_parameter].keys())

@st.fragment
def select_parameters_events(file_raw, raw_parameter, sub_parameter_list):
    with st.spinner('Selecting parameters...'):
        selected_events = select_events(root_files_list[0], loaded_root_file=file_raw, dataset='default', IS_DATA=False, list_of_event_data=sub_parameter_list)
    #st.write(selected_events)
    return selected_events

#if st.button('Update dataframe'):
selected_events = select_parameters_events(st.session_state.file_raw, raw_parameter, sub_parameter_list)

def filter_events(selected_events):
    my_dict = {}

    for col in selected_events.keys():
        #print(col)
        try: 
            my_dict[col] = ak.flatten(selected_events[col])
        except:
            my_dict[col] = ak.flatten(selected_events[col], axis=None)
    
    return my_dict


my_dict = filter_events(selected_events)
muon_df = pd.DataFrame.from_dict(my_dict)

st.dataframe(muon_df)

## Cut parameters
cut_parameter_value = {}
for parameter_cut in muon_df.columns:
    min_value = float(muon_df[parameter_cut].values.min())
    max_value = float(muon_df[parameter_cut].values.max())
    cut_parameter_value[parameter_cut] = st.slider(f"Cut for {parameter_cut}", min_value, max_value, value=(min_value, max_value), key=f'{parameter_cut}')

    cutted_dataframe = muon_df.loc[muon_df[parameter_cut].between(cut_parameter_value[parameter_cut][0],cut_parameter_value[parameter_cut][1] )]

#st.dataframe(cutted_dataframe)
#st.scatter_chart(data=cutted_dataframe) #x=None, y=None, x_label=None, y_label=None,)

@st.fragment
def plot_dataframe(dataframe):

    import plotly.express as px
    df = px.data.tips()
    nbins = st.slider('Number of bins:', 1, 500, 20)
    col_to_plot = st.selectbox('Column to plot:', cutted_dataframe.columns, index=1)
    fig = px.histogram(cutted_dataframe, x=col_to_plot, nbins=nbins)
    st.plotly_chart(fig)


plot_dataframe(cutted_dataframe)
#     pass
#sub_sub_parameter = st.selectbox(f'Parameters for {raw_parameter} > {sub_parameter}:',file_raw[raw_parameter][sub_parameter].keys())
#st.write(sub_sub_parameter)
# root_files_list


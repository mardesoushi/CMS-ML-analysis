import streamlit as st

###### From the notebook...
import logging
import awkward as ak
# from coffea import processor
# from coffea.nanoevents import NanoAODSchema
# from coffea.analysis_tools import PackedSelection

import matplotlib.pyplot as plt
import numpy as np

#import utils  # contains code for bookkeeping and cosmetics, as well as some boilerplate
import sys
sys.path.append('../')
from modules.prepare_data import *
##
# The classics
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# The newcomers
import awkward as ak
import uproot
import vector
vector.register_awkward()
import subprocess
import subprocess 
import numpy as np

# TODO: 
    # Apply Trigger selection, filter parameters by name
    # Create different dataframes


#file_example = 'https://opendata.cern.ch/record/30526/files/CMS_Run2016G_MET_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_270000_file_index.txt'
if 'root_files_list' not in st.session_state:
    st.session_state.root_files_list = []


st.session_state.file_id = st.text_input("""File ID""", value="30531", placeholder='File ID', help='Get the IDs from https://opendata.cern.ch/')
st.info('The data ID need to be from a nanoAOD format dataset.', icon='💡')
#st.session_state.file_link = st.text_input("""Link to ```.txt``` file""", value=file_example, placeholder=file_example)
#st.info(f'For example: \n\n ```{file_example}```')
@st.cache_data(show_spinner=False, persist='disk')
def download_data_file(open_data_file_id):
    ## Get a link to download .root files in .txt. format
    #open_data_file_id = st.session_state.file_id
    command = ["cernopendata-client", "get-file-locations", "--recid", open_data_file_id, "--protocol", "xrootd"]
    result = subprocess.run(command, capture_output=True, text=True)
    filenames = result.stdout.splitlines()
    filenames_sig = np.array(filenames)

    return filenames_sig

if st.button('Download data file'):
    with st.spinner('Downloading list of data files...'):
        root_files_list = download_data_file(st.session_state.file_id)
        st.session_state.root_files_list = root_files_list


if len(st.session_state.root_files_list) == 0:
    st.info("Select the first file from the list to get metadata")
    st.stop()


## Show downloaded files list
with st.expander('Root files list'):
    for file in st.session_state.root_files_list:
        st.text(file)
#st.write(root_files_list)


## Openfirst file to get metadata
if 'file_raw' not in st.session_state:
    st.write("Open first file from the list to get metadata:")
    with st.spinner(f'Opening file with uproot... File: {st.session_state.root_files_list[0]}'):
        try:
            file_raw = uproot.open(st.session_state.root_files_list[0])
            st.session_state.file_raw = file_raw
        except:
            print(f"Could not open {st.session_state.root_files_list[0]}")
            st.error(f"Could not open {st.session_state.root_files_list[0]}")



col1, col2 = st.columns(2)
events = st.session_state.file_raw['Events']
nevents = events.num_entries
st.write(f'Number of events in the file: {nevents}')
with col1:
    raw_parameter = st.selectbox('File raw parameters', st.session_state.file_raw.keys(), index=1 )

with col2:
    if raw_parameter == 'Events;1':
        sub_parameter_list = st.multiselect(f'Parameters for {raw_parameter}:' ,default=['MET_pt'], options=st.session_state.file_raw[raw_parameter].keys())

@st.fragment
def select_parameters_events(file_raw, raw_parameter, sub_parameter_list):
    with st.spinner('Selecting parameters...'):
        selected_events = select_events(st.session_state.root_files_list[0], loaded_root_file=file_raw, dataset='default', IS_DATA=False, list_of_event_data=sub_parameter_list)
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


col1, col2 = st.columns(2)
col1.dataframe(muon_df)

with col2:
    ## Apply HLT
    with st.container(height=400):
        HLT_filter = {}
        st.write('Apply HLT (preview):')
        for col in st.session_state.file_raw[raw_parameter].keys():
            if 'HLT' in col:
                HLT_filter[col] = st.checkbox(col, value=False, key=col)


# for hlt in HLT_filter.keys():
#     if 'HLT' in col:
#         muon_df = muon_df[muon_df[col] == HLT_filter[col]]
#st.write(['HLT' in x  for x in st.session_state.file_raw[raw_parameter].keys()]) # st.session_state.file_raw.keys())
# for col in st.session_state.file_raw.keys():
#     st.checkbox('HLT' )

@st.fragment
def plot_dataframe():

    cut_parameter_value = {}
    cutted_dataframe = muon_df.copy()
    for parameter_cut in muon_df.columns:
        min_value = float(muon_df[parameter_cut].values.min())
        max_value = float(muon_df[parameter_cut].values.max())
        cut_parameter_value[parameter_cut] = st.slider(f"Cut for {parameter_cut}", min_value, max_value, value=(min_value, max_value), key=f'{parameter_cut}')

        cutted_dataframe = cutted_dataframe.loc[cutted_dataframe[parameter_cut].between(cut_parameter_value[parameter_cut][0],cut_parameter_value[parameter_cut][1])].copy()
        #st.write(cut_parameter_value[parameter_cut])

    import plotly.express as px
    nbins = st.slider('Number of bins:', 1, 500, 20)
    col_to_plot = st.selectbox('Column to plot:', cutted_dataframe.columns, index=0)
    fig = px.histogram(cutted_dataframe, x=col_to_plot, nbins=nbins)
    st.plotly_chart(fig)
    #st.dataframe(cutted_dataframe)


plot_dataframe()


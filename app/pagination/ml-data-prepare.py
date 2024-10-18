import streamlit as st
import subprocess 
import numpy as np

import uproot
import matplotlib.pylab as plt
import awkward as ak
import numpy as np
from glob import glob
import pickle
DATA_PATH = "../data/"
 
col1, col2 = st.columns([3,1])
col1.title("ML Exercise: Anomaly detection in high energy physics")
col2.image("resources/front.png", width=200)

with st.expander("Description"):
    text_md = """
    In this we will demonstrate how to design a tiny autoencoder (AE) that we will use for anomaly detection in particle physics. More specifically, we will demonstrate how we can use autoencoders to select potentially New Physics enhanced proton collision events in a more unbiased way!

    We will train the autoencoder to learn to compress and decompress data, assuming that for highly anomalous events, the AE will fail.

    ## Dataset

    As a dataset, we will use the CMS Open data that you have been made familiar with already. Our dataset will be represented as an array of missing transverse energy (MET), up to 4 electrons, up to 4 muons and 10 jets each described by pT, η, φ and particle ID (just from knowing whether it is a muon/electron/jet)--recid 63168 --protocol xrootd. The particles are ordered by pT. If fewer objects are present, the event is zero padded.

    We will train on a QCD MC dataset (we could also train directly on data), and evaluate the AE performance on a New Physics simulated sample: A Bulk graviton decaying to two vector bosons: G(M=2 TeV) → WW

    We'll train using background data only and test using both background and the Graviton sample. Let's fetch them! The background data are available [here](https://opendata.cern.ch/record/63168) (recid = 63168) and the signal data [here](https://opendata.cern.ch/record/33703) (recid = 33703). The signal consists of 1,37M events and the background 19,279M events. We will use roughly 500K for each process.

    We will use the docker client to print all the file names. You can then use this list to concatenate data from all the files.
    """
    st.markdown(text_md, unsafe_allow_html=True)


    st.write("Based on: https://github.com/thaarres/qcd_school_ml")

st.divider()

@st.cache_data
def download_opendata_file(file_id):
    '''Get a link to download .root files in .txt. format from refid in CMS Open data'''
    command = ["cernopendata-client", "get-file-locations", "--recid", file_id, "--protocol", "xrootd"]
    result = subprocess.run(command, capture_output=True, text=True)
    filenames = result.stdout.splitlines()
    filenames = np.array(filenames)
    return filenames


with st.spinner('Fetching signal files...'):

    open_data_file_id_sig = st.text_input("File ID for signal", value="63168")
    filenames_sig = download_opendata_file(open_data_file_id_sig)


with st.spinner('Fetching background files...'):
    open_data_file_id_bkg = st.text_input("File ID for background", value="33703")
    filenames_bkg = download_opendata_file(open_data_file_id_bkg)
    


st.write('Number of background files: ' + str(len(filenames_bkg)))
st.write('Number of signal files: ' + str(len(filenames_sig)))

treename = "Events"
st.write("Treename: " + treename)

number_files_bkg = st.number_input("Number of background files:", value=1,  min_value=0, max_value=len(filenames_bkg))
#@st.cache_data(show_spinner=False)
def concatenate_data_using_uproot(filenames_sig, filenames_bkg, number_files_bkg):
    
    branch_dict = {
            "Muon": ["pt", "eta", "phi"],
            "Electron": ["pt", "eta", "phi"],
            "FatJet": ["pt", "eta", "phi"],
            # "MET": ["pt", "phi"]
        }
    # make list of branches to read from the dictionary above
    branch_names = []
    for obj, var in branch_dict.items(): 
        branch_names += [obj + "_" + v for v in var]

    infiles_sig = filenames_sig[:] # Lets use all the signal files

    ## aggregate all data files into one single python oobject
    data_sig = uproot.concatenate({fname:"Events" for fname in infiles_sig}, 
                                branch_names, 
                                how = "zip",
                                library = "ak")

    # infiles_bkg = ["root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/130000/2EDCC683-1B4B-614B-BEB7-D80BBC20AD8E.root","root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/270000/19E8D842-3175-1449-AF6C-FD9C69D12724.root","root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/270000/3957434B-7E09-3B4C-8329-FD44D82C7DB7.root","root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/270000/397D1673-167A-CF46-9E79-D7069D9AC359.root"]

    infiles_bkg = filenames_bkg[7: 2 + int(number_files_bkg)] #Let's use only 4 of the background files, one of the first files seems to not be accessible with xrootd so I take entry 4-8
    data_bkg = uproot.concatenate({fname:"Events" for fname in infiles_bkg}, 
                                branch_names, 
                                how = "zip",
                                library = "ak")
    # # Here is an example of how you can open a single file rith awkward and regex for the branch expression :

    # file_bkg = uproot.open(infiles_bkg[0])
    # data_bkg = file_bkg["Events"].arrays(
    #     filter_name = "/(Muon|Electron|FatJet|MET)_(pt|eta|phi|sumEt)/", 
    #     how = "zip"
    # )
    return data_sig, data_bkg

def save_data_to_pickle(data_sig, data_bkg):
    '''Converts data to list and then save data to pickle file'''
    signal_list = data_sig.to_list()
    background_list = data_bkg.to_list()

    # Salvar o array com pickle
    with open(DATA_PATH + "awk_signal_list.pkl", "wb") as f:
        pickle.dump(signal_list, f)

    with open(DATA_PATH + "awk_background_list.pkl", "wb") as f:
        pickle.dump(background_list, f)


@st.cache_data(show_spinner=False, persist='disk')
def read_data_from_pickle(sig_file_name = "awk_signal_list.pkl", bkg_file_name = "awk_background_list.pkl"):
    '''Read data from pickle file'''

    with open(DATA_PATH +  sig_file_name, "rb") as f:
        signal_list = pickle.load(f)

    with open(DATA_PATH +  bkg_file_name, "rb") as f:
        background_list = pickle.load(f)

    return signal_list, background_list


with st.spinner('Using uproot to concatenate signal and background files...'):
    try:
        signal_list, background_list = read_data_from_pickle(sig_file_name = "awk_signal_list.pkl", bkg_file_name = "awk_background_list.pkl")
        data_sig = ak.from_regular(signal_list)
        data_bkg = ak.from_regular(background_list)

    except:
        data_sig, data_bkg = concatenate_data_using_uproot(filenames_sig, filenames_bkg, number_files_bkg)
        save_data_to_pickle(data_sig, data_bkg)

st.divider()


with st.expander("Show histograms"):
    objects = [ 'Electron', 'FatJet', 'Muon']
    selected_objects = st.multiselect("Parameters for Muon_pt:", objects, default=objects)
    
    import mplhep
    mplhep.style.use('CMS')

    for obj in selected_objects:
        fig = plt.figure(figsize = (8,4))

        for label, data in zip(["bkg","sig"], [data_bkg, data_sig]):
            num = ak.num(data[obj]) ## convert akward object to numpy array

            # plot the histogram into "CMS" format
            plt.hist(num, label = label, bins = range(13), density = True, 
                    #log = True, 
                    histtype = "step")
            
        plt.xlabel(f"N of {obj}")
        plt.legend()
        st.pyplot(fig)




    for obj in selected_objects:
        fig = plt.figure(figsize = (8,4))

        for label, data in zip(["bkg","sig"], [data_bkg, data_sig]):
            # notice the [:,:1] below -> we slice the array and select no more than the first entry per event
            # ak.ravel makes the array flat such that we can fill a histogram
            plt.hist(ak.ravel(data[obj].pt[:,:1]), label = label, bins = np.linspace(1, 2000, 101), density = True, 
                    log = True, 
                    histtype = "step")
            
        plt.xlabel(f"{obj} pt")
        plt.legend()
        st.pyplot(fig)

    #     plt.grid()



### Format the data
def getPadNParr(events, obj, n_pad, fields, cuts = None, name = None, pad_val = 0):
    '''
    This function filter objects and pads them to a certain length with a given value
    events: events object, usually an 'akward' object
    obj: key of the object we are interested in - MET, Electron, Muon, FatJet, etc.
    n_pad: numbers of the same object in the same event, for the case where there are 2 muons, etc
    fields: list of physical proprieties of the object we want: ["pt", "eta", "phi", ...]
    cuts: cuts to be made to the arrays
    name: same as the key for the object -> MET, Electron, Muon, FatJet, etc.
    pad_val: value to be inserted in the None values of the ak.array, defaults to 0
    '''
    
    objects = events[obj]
    
    if not name: name = obj
    
    pad_arrs = []
    var_names = []
        
    # padding with nones
    pad_arr = ak.pad_none(objects, n_pad, clip=True)
    
    # combining to numpy
    for i in range(n_pad):

        for var in fields:
            pad_arrs += [ak.to_numpy( ak.fill_none(pad_arr[var][:,i], pad_val) )]
            var_names.append( "{}_{}_{}".format(name, i, var) )
            
    return np.stack(pad_arrs), var_names

def formatData(data, objects, verbosity = 0):
    '''
    This function concatenates the padded arrays for different objects.
    It is controlled via a dictionary as defined above

    data: events object, agreggated for all datasets, usually an 'akward' object
    objects: a list of dicts formated as  
        objects = [{"name" : "MET", "key" : "MET", "fields" : ["pt", "phi"], "n_obj" : 1 }, 
                    ...,
                    ]

    verbosity: controls the verbosity of the function, 0 for false, 1 for true

    '''
    
    # this will be filled by all required objects
    dataList = [] 
    varList = []
    
    # loop over the dicts on the object array
    for obj in objects: 
        print(obj)
        dat, names = getPadNParr(data, obj["key"], obj["n_obj"], obj["fields"], obj["cuts"] if "cuts" in obj else None, obj["name"] )
        dataList.append(dat)
        varList += names
        
    if verbosity > 0:
        print("The input variables are the following:")
        print(varList)
                
    # combining and returning (and transforming back so events are along the first axis...)
    return dataList, np.concatenate(dataList, axis = 0).T, varList

with st.spinner('Preparing data...'):
    objects = [
        # {"name" : "MET", "key" : "MET", "fields" : ["pt", "phi"], "n_obj" : 1 },
        {"name" : "FatJet", "key" : "FatJet", "fields" : ["pt", "eta", "phi"], "n_obj" : 6},
        {"name" : "Electron", "key" : "Electron", "fields" : ["pt", "eta", "phi"], "n_obj" : 4},
        {"name" : "Muon", "key" : "Muon", "fields" : ["pt", "eta", "phi"], "n_obj" : 4}
    ]
        

    dataList_, x_sig, var_names = formatData(data_sig, objects, verbosity = 99) 
    dataList, x_bkg, var_names = formatData(data_bkg, objects, verbosity = 0) 




# # Let's look at some of the inputs and see whether they make sense!

# for i,name in enumerate(var_names[:100]):
#     if "_3_" not in name: continue
        
#     plt.figure()
    
#     _ = plt.hist(x_bkg[:,i], bins = 50, log = True, density = True, label = "Bkg")
#     _ = plt.hist(x_sig[:,i], bins = _[1], histtype = "step", density = True, label = "Sig")
    
#     plt.xlabel(name)
#     plt.legend()
# #     break



### WRITE train and test data for later use
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
_ = scaler.fit(x_bkg)
x_bkg_scaled = scaler.transform(x_bkg)
x_sig_scaled = scaler.transform(x_sig)
    
        
# define training, test and validation datasets
X_train, X_test = train_test_split(x_bkg_scaled, test_size=0.2, shuffle=True)

print("Training data shape = ",X_train.shape)    
with h5py.File(DATA_PATH +'bkg_dataset.h5', 'w') as h5f:
    h5f.create_dataset('X_train', data = X_train)
    h5f.create_dataset('X_test', data = X_test)
    
with h5py.File(DATA_PATH +'signal_dataset.h5', 'w') as h5f2:
    h5f2.create_dataset('Data', data = x_sig_scaled)   

with h5py.File(DATA_PATH + 'bkg_scaler_dataset.h5', 'w') as h5f2:
    h5f2.create_dataset('bkg_scaler', data = x_bkg_scaled)       

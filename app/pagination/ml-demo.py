import streamlit as st

col1, col2 = st.columns([3,1])
col1.title("ML Exercise: Anomaly detection in high energy physics")
col2.image("resources/front.png", width=200)


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
st.write("Under construction...")
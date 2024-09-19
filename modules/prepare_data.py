
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


#########################################################################################
def build_lumi_mask(lumifile, tree, verbose=False):
    # lumifile should be the name/path of the file

    '''All CMS data is studied by data quality monitoring groups for various subdetectors to determine whether it is suitable for physics analysis. Data can be accepted or rejected in units of “luminosity sections”. 
    These are periods of time covering 2**18  LHC revolutions, or about 23 seconds. 
    The list of validated runs and luminosity sections is stored in a json file that can be downloaded from the Open Data Portal.
    First, obtain the file with the list of validated runs and luminosity sections for 2016 data:
    this file can be download from: wget https://opendata.cern.ch/record/14220/files/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'''
    import subprocess
    lumifile = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
    subprocess.run(["wget", f"https://opendata.cern.ch/record/14220/files/{file_name}"])

    good_luminosity_sections = ak.from_json(open(lumifile, 'rb'))

    # Pull out the good runs as integers
    good_runs = np.array(good_luminosity_sections.fields).astype(int)
    #good_runs

    # Get the good blocks as an awkward array
    # First loop over to get them as a list
    all_good_blocks = []
    for field in good_luminosity_sections.fields:
        all_good_blocks.append(good_luminosity_sections[field])

    # Turn the list into an awkward array
    all_good_blocks = ak.Array(all_good_blocks)
    all_good_blocks[11]

    # Assume that tree is a NanoAOD Events tree
    nevents = tree.num_entries
    if verbose:
        print(f"nevents: {nevents}")
        print()
        print("All good runs")
        print(good_runs)
        print()
        print("All good blocks")
        print(all_good_blocks)
        print()

    # Get the runs and luminosity blocks from the tree
    run = tree['run'].array()
    lumiBlock = tree['luminosityBlock'].array()

    if verbose:
        print("Runs from the tree")
        print(run)
        print()
        print("Luminosity blocks from the tree")
        print(lumiBlock)
        print()

    # ChatGPT helped me with this part!
    # Find index of values in arr2 if those values appear in arr1

    def find_indices(arr1, arr2):
        index_map = {value: index for index, value in enumerate(arr1)}
        return [index_map.get(value, -1) for value in arr2]

    # Get the indices that say where the good runs are in the lumi file
    # for the runs that appear in the tree
    good_runs_indices = find_indices(good_runs, run)

    # For each event, calculate the difference between the luminosity block for that event
    # and the good luminosity blocks for that run for that event
    diff = lumiBlock - all_good_blocks[good_runs_indices]

    if verbose:
        print("difference between event lumi blocks and the good lumi blocks")
        print(diff)
        print()

    # If the lumi block appears between any of those good block numbers, 
    # then one difference will be positive and the other will be negative
    # 
    # If it it outside of the range, both differences will be positive or 
    # both negative.
    #
    # The product will be negagive if the lumi block is in the range
    # and positive if it is not in the range
    prod_diff = ak.prod(diff, axis=2)

    if verbose:
        print("product of the differences")
        print(prod_diff)
        print()

    mask = ak.any(prod_diff<=0, axis=1)

    return mask


def select_events(filename: str = '', loaded_root_file = None, dataset='default', IS_DATA=False, list_of_event_data=[]):

    '''
    filename: the name of an input NanoAOD ROOT file. URL or location from the single .root file, to be accessed through uproot.open() method.
    dataset: string to be used to organize output files
    IS_DATA: a flag that should be set to `True` if the input datafile is *collision* data.


    This function saves 
    * Mass of the $t\overline{t}$ system
    * $p_T$ of the muon
    * $\eta$ of the muon
    * `pileup`
    * `weight`
    * `nevents`
    * `N_gen`
    * `gw_pos`
    * `gw_neg`

    This function will process a single root file into a .csv format with some key data extracted.
    It can be modified to extract different information from the root file as well.'''
    
    # print(f"Opening...{filename}")
    
    # try:
    #     f = uproot.open(filename)
    # except:
    #     print(f"Could not open {filename}")
    #     return None
    f = loaded_root_file
    events = f['Events']

    nevents = events.num_entries

    print(f"{nevents = }")

    # Get selected events as an array

    selected_events = {}
    for event_data in list_of_event_data:
        selected_events[event_data] = events[event_data].array()
                          

    # # MET ---------------------------------------------------------
    # met_pt = events['PuppiMET_pt'].array()
    #selected_events['PuppiMET_eta']  = 0*events['PuppiMET_pt'].array()  # Fix this to be 0
    # met_phi = events['PuppiMET_phi'].array() 

    #selected_events['ht_lep'] = selected_events['Muon_pt'] + selected_events['PuppiMET_pt']

    return selected_events


def apply_cuts(events, selected_events, cuts = True, IS_DATA=False):
    #####################################################################################
    # Cuts
    #####################################################################################
    cuts = {}
    if cuts:
        print("Applying cuts")
        # Particle-specific cuts --------------------------------------
        
        tau32 = selected_events['FatJet_tau3']/selected_events['FatJet_tau2']

        #cut_fatjet = (tau32>0.67) & (fatjet_eta>-2.4) & (fatjet_eta<2.4) & (fatjet_mSD>105) & (fatjet_mSD<220)
        cut_fatjet = (selected_events['FatJet_pt'] > 500) & (selected_events['FatJet_particleNet_TvsQCD'] > 0.5)

        cut_muon = (selected_events['Muon_pt'] > 55) & (selected_events['Muon_eta']>-2.4) & (selected_events['Muon_eta']<2.4) & \
                (selected_events['Muon_tightId'] == True) & (selected_events['Muon_miniIsoId']>1) & (selected_events['ht_lep']>150)

        cut_jet = (selected_events['Jet_btagDeepB'] > 0.5) & (selected_events['Jet_jetId']>=4)



        # Event cuts -------------------------------------------------
        cut_met = (selected_events['PuppiMET_pt'] > 50)

        cut_nmuons = ak.num(cut_muon[cut_muon]) == 1
        cut_njets = ak.num(cut_jet[cut_jet]) == 1


        cut_trigger = (events['HLT_TkMu50'].array())

        cut_ntop = ak.num(cut_fatjet[cut_fatjet]) == 1

        cut_full_event = None
        if IS_DATA:    
            mask_lumi = build_lumi_mask('Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt', events)#, verbose=True)
            cut_full_event = cut_trigger & cut_nmuons & cut_met & cut_ntop & mask_lumi
        else:
            cut_full_event = cut_trigger & cut_nmuons & cut_met & cut_ntop
        
        # Apply the cuts and calculate the di-top mass
        fatjets = ak.zip(
            {"pt": selected_events['FatJet_pt'][cut_full_event][cut_fatjet[cut_full_event]], 
            "eta": selected_events['FatJet_eta'][cut_full_event][cut_fatjet[cut_full_event]], 
            "phi": selected_events['FatJet_phi'][cut_full_event][cut_fatjet[cut_full_event]], 
            "mass": selected_events['FatJet_mass'][cut_full_event][cut_fatjet[cut_full_event]]},
            with_name="Momentum4D",
        )

        muons = ak.zip(
            {"pt": selected_events['Muon_pt'][cut_full_event][cut_muon[cut_full_event]], 
            "eta": selected_events['Muon_eta'][cut_full_event][cut_muon[cut_full_event]], 
            "phi": selected_events['Muon_phi'][cut_full_event][cut_muon[cut_full_event]], 
            "mass": selected_events['Muon_mass'][cut_full_event][cut_muon[cut_full_event]]},
            with_name="Momentum4D",
        )

        jets = ak.zip(
            {"pt": selected_events['Jet_pt'][cut_full_event][cut_jet[cut_full_event]], 
            "eta": selected_events['Jet_eta'][cut_full_event][cut_jet[cut_full_event]], 
            "phi": selected_events['Jet_phi'][cut_full_event][cut_jet[cut_full_event]], 
            "mass": selected_events['Jet_mass'][cut_full_event][cut_jet[cut_full_event]]},
            with_name="Momentum4D",
        )

        met = ak.zip(
            {"pt": selected_events['PuppiMET_pt'][cut_full_event], 
            "eta": selected_events['PuppiMET_eta'][cut_full_event], 
            "phi": selected_events['PuppiMET_phi'][cut_full_event], 
            "mass": 0}, # We assume this is a neutrino with 0 mass
            with_name="Momentum4D",
        )

        p4mu, p4fj, p4j, p4met = ak.unzip(ak.cartesian([muons, fatjets, jets, met]))
        
        p4tot = p4mu + p4fj + p4j + p4met
        
        # Shape the weights and pileup
        N_gen = -999
        pileup = -999
        gw_pos = -999
        gw_neg = -999

        pileup_per_candidate = None
        
        tmpval_events = np.ones(len(ak.flatten(p4tot.mass)))
        tmpval = ak.ones_like(p4tot.mass)


        # Put in the MC weights
        if not IS_DATA:
            gen_weights = events['genWeight'].array()[cut_full_event]
            pileup = events['Pileup_nTrueInt'].array()[cut_full_event]

            gen_weights_per_candidate = tmpval * gen_weights
            #print(gen_weights_per_candidate)

            pileup_per_candidate = tmpval * pileup
            #print(pileup_per_candidate)

            # Get values associated with the total number of events. 
            # It's going to duplicate the number of entries, but we'll save the same value to 
            # each event
            gen_weights_org = events['genWeight'].array()

            gw_pos = ak.count(gen_weights_org[gen_weights_org > 0])
            gw_neg = ak.count(gen_weights_org[gen_weights_org < 0])
            N_gen = gw_pos - gw_neg
        else:
            pileup_per_candidate = -999*tmpval
            gen_weights_per_candidate = -999*tmpval
        

        # Build a dictionary and dataframe to write out the subset of data
        # we are interested in
        mydict = {}
        mydict['mtt'] = ak.flatten(p4tot.mass) 
        mydict['mu_pt'] = ak.flatten(p4mu.pt) 
        mydict['mu_abseta'] = np.abs(ak.flatten(p4mu.eta))
        mydict['pileup'] = ak.flatten(pileup_per_candidate)
        mydict['weight'] = ak.flatten(gen_weights_per_candidate)
        mydict['nevents'] = nevents*tmpval_events
        mydict['N_gen'] = N_gen*tmpval_events
        mydict['gw_pos'] = gw_pos*tmpval_events
        mydict['gw_neg'] = gw_neg*tmpval_events

        df = pd.DataFrame.from_dict(mydict)

        outfilename = f"OUTPUT_{dataset}_{filename.split('/')[-1].split('.')[0]}.csv"
        print(f'Saving output to {outfilename}')

        df.to_csv(outfilename, index=False)

        return df
import pickle
import numpy as np
import pandas as pd

import formatCompetitionDataHelpers25 as h2

# General parameters
logBoth = True # Whether to log-normalize input features
output_type = 'phoneme'

# Set up file paths:
root = '/home/towneib/ece553_reproducibility_challenge'

dataDir = root + '/brain2text25/hdf5_data_final' # Directly downloaded from Dryad
if logBoth:
        dataSave = root + '/brain2text25/data_log_both'
else:
        dataSave = root + '/brain2text25/data'


# Get list of session names corresponding to days with Switchboard data

# Load in CSV of all session/block information
sessiondat = pd.read_csv(root + '/brain2text25/t15_copyTaskData_description.csv')

# Pop the "Post-implant day" key because we don't actually need that
sessiondat.pop('Post-implant day', None)

# Remove all entries where the Corpus isn't Switchboard
switchboard_mask = sessiondat['Corpus'] == 'Switchboard'

onlysb = {}
for key in sessiondat.keys():
     # Change from pandas series to list for later ease-of-use
    onlysb[key] = sessiondat[key][switchboard_mask].tolist()

trainblocks = {}
valtestblocks = {}
# Get the specific blocks to use for each day corresponding to switchboard data
# This is necessary because multiple different corpus can be run in a single session
for idx in range(0, len(onlysb['Date'])): # Iterate over the dates with switchboard data

    if(onlysb['Split'][idx] == 'Train'): # Check the split
        if(onlysb['Date'][idx] not in trainblocks.keys()): # Add day to trainblocks if not already there
            trainblocks[onlysb['Date'][idx]] = []

        trainblocks[onlysb['Date'][idx]].append(onlysb['Block number'][idx]) # Add current block to the list of blocks to use for the current day

    elif(onlysb['Split'][idx] == 'Val/Test'): # Check the split
        if(onlysb['Date'][idx] not in valtestblocks.keys()): # Add day to valtestblocks if not already there
            valtestblocks[onlysb['Date'][idx]] = []

        valtestblocks[onlysb['Date'][idx]].append(onlysb['Block number'][idx]) # Add current block to the list of blocks to use



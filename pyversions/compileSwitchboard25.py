import pickle
import numpy as np
import pandas as pd

import formatCompetitionDataHelpers25 as h2

# General parameters
logBoth = True # Whether to log-normalize input features
output_type = 'phoneme'

# TODO: Download t15_copyTask_neuralData.zip from https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85#citations
# Extract to a folder of your choice, make sure dataDir points to the hdf5_data_final folder inside of that zip file

# Set up file paths:
root = '/home/towneib/ece553_reproducibility_challenge'

dataDir = root + '/brain2text25/hdf5_data_final' # Directly downloaded from Dryad
if logBoth:
        dataSave = root + '/brain2text25/data_log_both/allDatasets_log_both_no_8-13-23.pkl'
else:
        dataSave = root + '/brain2text25/data/allDatasets_no_log.pkl'


# Get list of session names corresponding to days with Switchboard data

# Load in CSV of all session/block information
sessiondat = pd.read_csv(root + '/brain2text25/t15_copyTaskData_description.csv')

# Pop the "Post-implant day" key because we don't actually need that
sessiondat.pop('Post-implant day')

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

# Now, start compiling the data sets (similar to some of the code in formatCompetitionData.ipynb)

# TEMP - Remove 8-13-2023 because train and test was throwing runtime warnings for invalid values in log1p and log (for the test it was a divide by zero in log)
if(logBoth == True):
    trainblocks.pop('2023-08-13')
    valtestblocks.pop('2023-08-13')

trainDatasets = [] # Train
testDatasets = []  # Validation (I know the variable says "test", but this is just following the author's and competition's convention)
competitionDatasets = [] # Test
for date in np.unique(onlysb['Date']):
    if date in trainblocks.keys():
        traindataset = h2.getDataset(dataDir + '/t15.' + date.replace("-", ".") + '/data_train.hdf5', trainblocks[date], logBoth, isHoldout=False)
        trainDatasets.append(traindataset)
        print("Adding " + date + " blocks " + np.array2string(np.unique(traindataset['blockNums']), separator = ", ") + " to training set")
    
    if date in valtestblocks.keys():
        testDataset = h2.getDataset(dataDir + '/t15.' + date.replace("-", ".") + '/data_val.hdf5', valtestblocks[date], logBoth, isHoldout=False)
        testDatasets.append(testDataset)
        print("Adding " + date + " blocks " + np.array2string(np.unique(testDataset['blockNums']), separator = ", ") + " to test (validation) set")

        competitionDataset = h2.getDataset(dataDir + '/t15.' + date.replace("-", ".") + '/data_test.hdf5', valtestblocks[date], logBoth, isHoldout=True)
        competitionDatasets.append(competitionDataset)
        print("Adding " + date + " blocks " + np.array2string(np.unique(competitionDataset['blockNums']), separator = ", ") + " to competition (test) set")

# Put everything into a single file and save like the author's code does
allDatasets = {}
allDatasets['train'] = trainDatasets
allDatasets['test'] = testDatasets
allDatasets['competition'] = competitionDatasets
with open(dataSave, 'wb') as handle:
     pickle.dump(allDatasets, handle)

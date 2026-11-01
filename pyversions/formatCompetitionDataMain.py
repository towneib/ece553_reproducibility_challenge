import pickle
import os
import re
import scipy
import numpy as np

import formatCompetitionDataHelpers as h

# Needed for G2P package to work (graphene to phoneme)
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')


"""
Set data and save directories
"""

server = 'towneib' # Current user
year = '24' # Year of the competition
logBoth = False # Whether you are planning to save log-normalized data
output_type = 'phoneme' # What is output by the transformer. Other options are 'char' and 'both'

if server == 'towneib':
    if year == '24':
        dataDir = '/home/towneib/ece553_reproducibility_challenge/brain2text24/competitionData' # Directly downloaded from Dryad
        if logBoth:
            dataSave = '/home/towneib/ece553_reproducibility_challenge/brain2text24/data_log_both'
        else:
            dataSave = '/home/towneib/ece553_reproducibility_challenge/brain2text24/data'
    elif year == '25': # NOT FULLY IMPLEMENTED
        dataDir = '/home/towneib/ece553_reproducibility_challenge/brain2text25/hdf5_data_final' # Directly downloaded from Dryad
        if logBoth:
                dataSave = '/home/towneib/ece553_reproducibility_challenge/brain2text25/data_log_both'
        else:
                dataSave = '/home/towneib/ece553_reproducibility_challenge/brain2text25/data'

if output_type == 'char':
    dataSave = f"{dataSave}_char"
    
if output_type == 'both':
    dataSave = f"{dataSave}_char_and_phoneme"
    
print("=========== CHOSEN DIRECTORIES ===========")
print("Data: " + dataDir + "\nSave: " + dataSave + "\n")

"""
Define list of session names (from competition data)
"""
sessionNames = ['t12.2022.04.28',  't12.2022.05.26',  't12.2022.06.21',  't12.2022.07.21',  't12.2022.08.13',
't12.2022.05.05',  't12.2022.06.02',  't12.2022.06.23',  't12.2022.07.27',  't12.2022.08.18',
't12.2022.05.17',  't12.2022.06.07',  't12.2022.06.28',  't12.2022.07.29',  't12.2022.08.23',
't12.2022.05.19',  't12.2022.06.14',  't12.2022.07.05',  't12.2022.08.02',  't12.2022.08.25',
't12.2022.05.24',  't12.2022.06.16',  't12.2022.07.14',  't12.2022.08.11']
sessionNames.sort()

#test = scipy.io.loadmat(dataDir + '/train/' + sessionNames[0] + '.mat')

"""
Keys in a single one of the training .mat files:
- 'sentenceText'
- 'tx1'
- 'tx2'
- 'tx3'
- 'tx4'
- 'spikePow'
- 'blockIdx'

"""

"""
======================== HOLDOUT DATA TESTING ========================
"""
trainDatasets = []
testDatasets = []
competitionDatasets = []

condition = 0

if condition == 0:
    evaluation_days = ['07.21', '07.27', '08.02', '08.11', '08.13'] 
if condition == 1:
    evaluation_days = ['06.28', '07.05', '07.14', '07.21', '07.27', '08.02', '08.11', '08.13'] 
    
skip_days = ['07.29']

trainCount = 0
evalCount = 0
print(logBoth)
for dayIdx in range(len(sessionNames[:-3])):
    
    if sessionNames[dayIdx][-5:] in skip_days:
        print("Skipping day: ", sessionNames[dayIdx][-5:])
        continue
        
    if sessionNames[dayIdx][-5:] in evaluation_days:
        
        evalCount += 1
        
        print(f"Placing {sessionNames[dayIdx]} in test/comp")
        
        # put both train and test data for held-out days into "test (really val)".
        testDataset1 = h.getDataset(dataDir + '/train/' + sessionNames[dayIdx] + '.mat', logBoth)
        testDataset2 = h.getDataset(dataDir + '/test/' + sessionNames[dayIdx] + '.mat', logBoth)
        testDataset = h.combine_dicts(testDataset1, testDataset2)
        testDatasets.append(testDataset)
        
        # only testing on these held-out days now. 
        if os.path.exists(dataDir + '/competitionHoldOut/' + sessionNames[dayIdx] + '.mat'):
            dataset = h.getDataset(dataDir + '/competitionHoldOut/' + sessionNames[dayIdx] + '.mat', logBoth)
            competitionDatasets.append(dataset)
            
    else:
        
        trainCount += 1
        
        print(f"Train {sessionNames[dayIdx]} in train")
        
        # put both train and test data for non held-out days into train
        trainDataset1 = h.getDataset(dataDir + '/train/' + sessionNames[dayIdx] + '.mat', logBoth)
        trainDataset2 = h.getDataset(dataDir + '/test/' + sessionNames[dayIdx] + '.mat', logBoth)
        trainDataset = h.combine_dicts(trainDataset1, trainDataset2) 
        
        # They use k-fold cross validation, so combining the train and test data makes sense
        trainDatasets.append(trainDataset)
        
        
print("Training days: ", trainCount)
print("Eval days: ", evalCount)





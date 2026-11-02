import pickle
import os
import re
import scipy
import numpy as np
from g2p_en import G2p

# From the Brain2Text'25 competition page/corresponding github
import h5py

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data


def combine_dicts(dict1, dict2):
    """
    Combines two dictionaries such that the data from both dictionaries resides in a single one, with there being no way to determine which data came from which input dictionary
    """
    combined_dict = {}
    for key in dict1.keys():
        if key in ['sentenceDat', 'transcriptions', 'phonemes']: # These keys are lists
            combined_dict[key] = dict1[key] + dict2[key] # Concatenation for lists
        else:
            combined_dict[key] = np.concatenate([dict1[key], dict2[key]]) # The other keys are numpy arrays, so use np.concatenate to combine them
        
    return combined_dict

def partition_by_blocks(dataset, train_blocks, test_blocks):
    
    """Split dataset into train and test portions using blockNums."""
    train_set, test_set = {}, {}
    
    for key, values in dataset.items():
        # blockNums is special: we need it to decide the split
        if key == "blockNums":
            continue

        # Assume dataset["blockNums"] is aligned with all other keys
        train_set[key] = [v for v, b in zip(values, dataset["blockNums"]) if b in train_blocks]
        test_set[key]  = [v for v, b in zip(values, dataset["blockNums"]) if b in test_blocks]

    # Also save the blockNums
    train_set["blockNums"] = [b for b in dataset["blockNums"] if b in train_blocks]
    test_set["blockNums"]  = [b for b in dataset["blockNums"] if b in test_blocks]

    return train_set, test_set

def return_train_validation_blocks(unique_blocks, K, fold_number):
    
    """
    list unique_blocks: unique blocks in that day
    int K: the number of folds
    fold_number: current fold
    
    Returns the blocks that should be used for training and validation. 
    
    * This function assumes the last block was already used for validation, since this was the default competition settings.
    """
    
    n_blocks_for_val = len(unique_blocks) - 1
    n_blocks_per_fold = np.ceil(n_blocks_for_val / K)
    
    start_idx = int(fold_number*n_blocks_per_fold)
    end_idx = int(min(n_blocks_for_val, (fold_number+1)*n_blocks_per_fold))
        
    blocks_for_val = unique_blocks[start_idx:end_idx]
    
    return np.setdiff1d(unique_blocks, blocks_for_val), blocks_for_val

g2p = G2p()
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']

def phoneToId(p):
    return PHONE_DEF_SIL.index(p)
CHAR_VOCAB = [
    "<sp>",          # space token
    "!", ",", ".", "?", "'",   # punctuation (incl. apostrophe)
] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 'a'..'z'

# Build mappings
_CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
_ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}

# Convenience indices
SPACE_ID = _CHAR_TO_ID["<sp>"]

def charToId(c: str) -> int:
    """Map raw input char to ID, normalizing space and lowercase."""
    if c == " ":
        c = "<sp>"
    c = c.lower()
    return _CHAR_TO_ID[c]

def idToChar(i: int) -> str:
    return _ID_TO_CHAR[i]

def reorder_electrode_idxs(feats):
    
    '''
    In the original array, first 128 indices of feats are txcrossings, 
    second 128 features are spikeband power.
    Here what we do is rearrange feats so that:
        0-64: tx for inferior
        64-128: spikepow for inferior
        128-194: tx for superior
        194-256: spikepow for superior
    '''
    
    area_6v_superior = np.array([
    [62,  51,  43,  35,  94,  87,  79,  78],
    [60,  53,  41,  33,  95,  86,  77,  76],
    [63,  54,  47,  44,  93,  84,  75,  74],
    [58,  55,  48,  40,  92,  85,  73,  72],
    [59,  45,  46,  38,  91,  82,  71,  70],
    [61,  49,  42,  36,  90,  83,  69,  68],
    [56,  52,  39,  34,  89,  81,  67,  66],
    [57,  50,  37,  32,  88,  80,  65,  64]
    ])
    
    superior_combined_array = np.empty(area_6v_superior.size * 2, dtype=area_6v_superior.dtype)  
    
    # first 64 features are tx, second 64 are spikeband power
    superior_combined_array[:64] = area_6v_superior.ravel()
    superior_combined_array[64:] = area_6v_superior.ravel() + 128   

    area_6v_inferior = np.array([
        [125, 126, 112, 103,  31,  28,  11,  8],
        [123, 124, 110, 102,  29,  26,   9,  5],
        [121, 122, 109, 101,  27,  19,  18,  4],
        [119, 120, 108, 100,  25,  15,  12,  6],
        [117, 118, 107,  99,  23,  13,  10,  3],
        [115, 116, 106,  97,  21,  20,   7,  2],
        [113, 114, 105,  98,  17,  24,  14,  0],
        [127, 111, 104,  96,  30,  22,  16,  1]
    ])
    
    inferior_combined_array = np.empty(area_6v_inferior.size * 2, dtype=area_6v_inferior.dtype)  
    inferior_combined_array[:64] = area_6v_inferior.ravel()
    inferior_combined_array[64:] = area_6v_inferior.ravel() + 128   
    
    feats_reshaped = np.zeros_like(feats)
    
    feats_reshaped[:, 0:128] = feats[:, inferior_combined_array.ravel()]
    feats_reshaped[:, 128:] = feats[:, superior_combined_array.ravel()]
    
    return feats_reshaped
    
"""
For now, I am NOT implementing the logBoth thing. I want data first. Then I can make things fancy
"""
def loadFeaturesAndNormalize(sessionPath, logBoth):

    # Load data from .mat file
    dat = load_h5py_file(sessionPath)

    # Set up new lists to be added to the data structure
    input_features = []
    transcriptions = []
    frame_lens = []
    block_means = []
    block_stds = []
    n_trials = len(dat['sentence_label']) # It is a list instead of a numpy array in the '25 version
    #Number of sentences = number of trials

    # The 25' data comes with a lot of these things already, so I just need to pick the electrodes I want
    # for the features and put things from existing matrices and lists into the ones the '24 code has defined
    # so it can be reused (even if the implementation is super inefficient)
    #collect area 6v tx1 and spikePow features
    for i in range(n_trials):    
        # #get time series of TX and spike power for this trial
        # #first 128 columns = area 6v only
        # if logBoth:
        #     tx_crossings = np.log1p(dat['tx1'][0, i][:, 0:128])
        #     log_pow = np.log(dat['spikePow'][0,i][:,0:128])
        #     features = np.concatenate([tx_crossings, log_pow], axis=1)
        # else:
        #     # tx1 and spikePow are nested weird when it comes out of the .mat file, so the [0, i] is to index the 1x1 array that the actual data is in and access the data for the ith trial
        #     # The [:, 0:128] is to then take the first 128 columns from the TxF matrix of either threshold crossing rate or spike band power (T = number of bins, F = number of channels)
        #     # T can change (and does) for each trial, as each trial has a different length and thus a different number of 20 ms bins in it
        #     # Axis=1 concatenates columns, so you end up with a TxF matrix where columns 0-127 are tx1 and columns 128-255 are spikePow
        #     features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

        #sentence_len = features.shape[0] # Number of time bins for a specific trial = T = number of rows in features
        #sentence = dat['sentence_label'][i].strip() # Strip removes white space before and after a sentence

        features = np.concatenate([dat['neural_features'][i][:, 0:64], dat['neural_features'][i][:, 256:320], dat['neural_features'][i][:, 192:256], dat['neural_features'][i][:, 448:512]])

        input_features.append(features)
        transcriptions.append(dat['sentence_label'][i].strip())
        #frame_lens.append(sentence_len) ==> This is just the n_time_steps field of dat

    #frame_lens = dat['n_time_steps']

    # At this point we have:
    # input_features: a list of length n_trials whose entries are the TxF arrays of features for each trial
    # transcriptions: the "sentence" key from the original data put into a normal list of length n_trials. Each entry is the sentence for the ith trial
    # frame_lens: a list of length n_trials with the value of T for each trial, where T is the number of 20ms time bins in that trial

    # block-wise feature normalization ==> According to the readme, the data was recorded in blocks that can be used to 
    # "implement blockwise feature normalization (e.g., z-scoring), which helps combat feature nonstationarity."
    blockNums = dat['block_num'] #np.squeeze(dat['blockIdx']) # Turns the nested array structure from the original data field into a 1d array (squeeze removes dimensions of size 1)
    blockList = np.unique(blockNums) # Take the unique values from the list to get the total number of recording blocks within the session
    blocks = []

    # Sorts the data into blocks by, for each unique block number, getting the indices for the feature arrays and such
    # that have that specific block label.
    # I'm assuming that the type casting from the output of np.argwhere (where the entries are all int64s) to int32s is because
    # later code breaks or something if int64s are used.
    # The output is blocks, a list of length (number of unique blocks) where each entry is an array of the corresponding trials
    # that were recorded in that block. This will be used for blockwise featured normalization.
    # EACH BLOCK SEEMS TO HAVE A MAXIMUM OF TWENTY SENTENCES IN IT ==> For the '25 set, each block as 50 sentences, NOT 20 like the '24 set did
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums==blockList[b])
        sentIdx = sentIdx[:,0].astype(np.int32)
        blocks.append(sentIdx)

    # Does the blockwise feature normalization by z-scoring the data in each block
    for b in range(len(blocks)):
        # The indexing of input_features is meant to take the TxF matrices for the trials in block b and concatenate them vertically (stacked on each other)
        # ==> feats ends up being sum(Ts for trials in block b)xF
        # The blocks[b][0] gets the index of the first trial in the bth block
        # The blocks[b][-1] + 1 gets the index of the last trial in the bth block. The +1 is because python indexing is [inclusive:exclusive]
        # The reorder_electrode_idxs is so the feature means and stds (which are vectors) are substracted from the right features later on (I think
        # that if they just did "reorder_electrode_idxs" after everything instead of running it twice, they would get the exact same result)
        
        # ==> I already did the reordering (it is currently hard-coded in)
        # feats = reorder_electrode_idxs(np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0))
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)

        feats_mean = np.mean(feats, axis=0, keepdims=True)  # Calculate the mean of all of the features (axis = 0 means you are taking the mean of each row, which is a vector of length F)
        feats_std = np.std(feats, axis=0, keepdims=True) # Same as np.mean but for the standard deviation
        for i in blocks[b]: # For the trial indices in the bth block
            #input_features[i] = reorder_electrode_idxs(input_features[i]) # Do the idx reordering
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8) # Z-score normalization by the feature means and stds of all trials in the bth block, 1e-8 prevents divide by 0

    #convert to tfRecord file ==> Zero idea as to what this means
    session_data = { # Put the new reformatted things into a struct and return
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': np.array(dat['n_time_steps']), 
        'blockNums': blockNums,
        'text': dat['seq_class_ids'],
        'textLens': np.array(dat['seq_len']),

        
    }

    return session_data


def getDataset(fileName, logBoth, output_type="phoneme"):
    session_data = loadFeaturesAndNormalize(fileName, logBoth)

    allDat = []
    trueSentences = []
    seq_char = []     # for characters (primary text)
    seq_phone = []    # for phonemes  (secondary text2, only when output_type == 'both')
    # Iterate over every trial
    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x]) # Add TxF feature matrix for xth trial to list of feature matrices for all of the data in this session
        trueSentences.append(session_data['transcriptions'][x]) # Same but adding all of the sentences to a list of all trial sentences in this session

        # clean sentence
        thisTranscription = str(session_data['transcriptions'][x]).strip() # I think this is redundant because they already do it in loadFeaturesandNormalize
        thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription) # Removes anything from the sentence that isn't a letter, hyphen, space, or apostraphe (thanks Copilot)
        thisTranscription = thisTranscription.replace('--', '').lower() # Replaces '--' with '' (not sure why this wasn't in the re.sub, maybe too complicated?) and turns sentence to lowercase 
        addInterWordSymbol = True
        maxSeqLen = 500

        # ----- build phoneme ids (used for 'phoneme' or 'both') ----- ==> See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for explanations of phonemes
        # if output_type in ('phoneme', 'both'):
        #     phonemes = []
        #     for p in g2p(thisTranscription):
        #         if addInterWordSymbol and p == ' ':
        #             phonemes.append('SIL') # Adds in "SILENCE" as a phoneme (literally denotes silence)
        #         p = re.sub(r'[0-9]', '', p)           # remove stress ==> g2p returns things like AA0 and AA1, which are both an "AA" phoneme but with the stress in different places
        #         if re.match(r'[A-Z]+', p):            # keep only phoneme labels
        #             phonemes.append(p)
        #     if addInterWordSymbol:
        #         phonemes.append('SIL') # Adds "SILENCE" after the end of the sentence, maybe to make it so after every word there is a "SIL"?

        #     seqLenP = len(phonemes) # Number of phonemes (including SIL if addInterWordSymbol=true) that make up the current sentence
        #     seqIDsP = np.zeros([maxSeqLen], dtype=np.int32) # Create a placeholder array that the phoneme ID will be put in ==> Off of the PHONE_DEF_SIL matrix they define
        #     seqIDsP[:seqLenP] = [phoneToId(p) + 1 for p in phonemes] # +1 is so IDs are 1 indexed ==> This array has a TON of zeros in it if seqLenP < maxSeqLen

        # # ----- build char ids (used for 'char' or 'both') -----
        # if output_type in ('char', 'both'):
        #     txt = thisTranscription.replace('-', ' ')
        #     characters = []
        #     for c in txt:
        #         if addInterWordSymbol and c == ' ':
        #             characters.append('<sp>')
        #         else:
        #             characters.append(c)
        #     if addInterWordSymbol:
        #         characters.append('<sp>')

        #     seqLenC = len(characters)
        #     seqIDsC = np.zeros([maxSeqLen], dtype=np.int32)
        #     try:
        #         seqIDsC[:seqLenC] = [charToId(c) + 1 for c in characters]
        #     except Exception:
        #         print(txt)

        # ----- save according to output_type -----
        if output_type == 'phoneme':
            seq_char.append(session_data['text'][x])          # keep same key names as original ('text' holds the chosen output) ==> They use seq_char because that is what is put into newDataset
        # elif output_type == 'char':
        #     seq_char.append(seqIDsC)
        # elif output_type == 'both':
        #     seq_char.append(seqIDsC)          # 'text'  -> characters
        #     seq_phone.append(seqIDsP)         # 'text2' -> phonemes ==> Added later if output is both
        # else:
        #     print("invalid output type, must be 'phoneme', 'char', or 'both'")
        #     return 0

    # ----- package dataset -----
    newDataset = {
        'sentenceDat': allDat, # List of all re-ordered feature matrices (size TxF), length = n_trials ==> ONLY OF tx1 and spikePow for the 128 electrodes in 6v!!!!
        'transcriptions': trueSentences, # List of the sentences used in each trial, length = n_trials
        'text': seq_char,                     # primary (chars for 'both') ==> List of all the phoneme IDs corresponding to the PHONE_DEF_SIL matrix defined in the code, length = n_trials
        'blockNums': session_data['blockNums'], # length = n_trials list containing the recording block each trial belongs to (no modifications have been made to this list)
        'textLens': session_data['textLens'],
        'timeSeriesLens': session_data['frameLens']
    }

    # # lengths for primary text
    # timeSeriesLens = [arr.shape[0] for arr in newDataset['sentenceDat']] # list of length = n_trials of the value of T (number of time bins) in each trial
    # textLens = []
    # for arr in newDataset['text']: # Iterates over each element in 'text' = number of phonemes in each trial sentence if output = phoneme (includes SIL if addInterWordSymbol = true)
    #     zeroIdx = np.argwhere(arr == 0) # Finds all of the indices were the value is 0 ==> gives a nested array, so you need two indices
    #     textLens.append(zeroIdx[0, 0] if zeroIdx.size else arr.shape[0]) # zeroIdx[0, 0] = index where the first 0 is = length of the sentence in phonemes (zeroIdx.size==0 means length = maxSeqLen, so just use arr.shape[0])

    # newDataset['timeSeriesLens'] = np.array(timeSeriesLens)
    # newDataset['textLens'] = np.array(textLens)
    newDataset['textPerTime'] = newDataset['textLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32) # For each trial, divide number of phonemes in sentence by number of time bins for trial to get phonemes per time bin

    # add phoneme versions when requested
    if output_type == 'both':
        newDataset['text2'] = seq_phone
        textLens2 = []
        for arr in newDataset['text2']:
            zeroIdx = np.argwhere(arr == 0)
            textLens2.append(zeroIdx[0, 0] if zeroIdx.size else arr.shape[0])
        newDataset['textLens2'] = np.array(textLens2)
        newDataset['textPerTime2'] = newDataset['textLens2'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)

    return newDataset


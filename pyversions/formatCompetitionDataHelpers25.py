import re
import numpy as np
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
    
# Take in path to h5py file, list of blocks to include in loaded features, whether to log-normalize features, if data is heldout
# The last param is important because the heldout data has a ton of parameters filled with arrays of 'None' which REALLY breaks things
def loadFeaturesAndNormalize(sessionPath, blocklist, logBoth, isHoldout):

    # Load data from .mat file
    dat = load_h5py_file(sessionPath)

    # Take out the blocks we don't want
    blockmask = [dat['block_num'][i] in blocklist for i in range(0, len(dat['block_num']))]
    for key in dat.keys():
        dat[key] = [dat[key][x] for x in range(0,len(dat[key])) if blockmask[x] == True]
    # Set up new lists to be added to the data structure
    input_features = []
    transcriptions = []

    n_trials = len(dat['sentence_label']) # It is a list instead of a numpy array in the '25 version
    #Number of sentences = number of trials

    # The 25' data comes with a lot of these things already, so I just need to pick the electrodes I want
    # for the features and put things from existing matrices and lists into the ones the '24 code has defined
    # so it can be reused (even if the implementation is super inefficient)
    #collect area 6v tx1 and spikePow features
    for i in range(n_trials):    
        
        # Order: 0-63 ==> Ventral 6v Tx, 64-127 ==> Ventral 6v SBP, 128-191 ==> Dorsal 6v Tx, 192-255 ==> Dorsal 6v SBP
        if logBoth:
            # The original code took log1p of Tx and normal log of SBPs
            features = np.concatenate([np.log1p(dat['neural_features'][i][:, 0:64]), np.log(dat['neural_features'][i][:, 256:320]), np.log1p(dat['neural_features'][i][:, 192:256]), np.log(dat['neural_features'][i][:, 448:512])], axis = 1)
        else:
            features = np.concatenate([dat['neural_features'][i][:, 0:64], dat['neural_features'][i][:, 256:320], dat['neural_features'][i][:, 192:256], dat['neural_features'][i][:, 448:512]], axis = 1)

        input_features.append(features)
        
        # Adding try catch so when the heldout data is run, which has arrays filled with 'None', this line doesn't fail
        if not isHoldout:
            transcriptions.append(dat['sentence_label'][i].strip())
        else:
            transcriptions.append('HELD OUT') # This is what the '24 holdout data has in it
            # Also handle 
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
    # EACH BLOCK SEEMS TO HAVE A MAXIMUM OF TWENTY SENTENCES IN IT ==> For the '25 set, each block as a maximum of 50 sentences, NOT 20 like the '24 set did
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
        'text': dat['seq_class_ids'] if not isHoldout else [np.zeros(500) for x in range(0, n_trials)],
        'textLens': np.array(dat['seq_len']) if not isHoldout else 8.*np.ones(np.array(dat['seq_len']).shape)

        
    }

    return session_data


def getDataset(fileName, blocklist, logBoth, isHoldout, output_type="phoneme"):
    session_data = loadFeaturesAndNormalize(fileName, blocklist, logBoth, isHoldout)

    allDat = []
    trueSentences = []
    seq_char = []     # for characters (primary text)
    seq_phone = []    # for phonemes  (secondary text2, only when output_type == 'both')
    # Iterate over every trial
    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x]) # Add TxF feature matrix for xth trial to list of feature matrices for all of the data in this session
        trueSentences.append(session_data['transcriptions'][x]) # Same but adding all of the sentences to a list of all trial sentences in this session

        # clean sentence
        # thisTranscription = str(session_data['transcriptions'][x]).strip() # I think this is redundant because they already do it in loadFeaturesandNormalize
        # thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription) # Removes anything from the sentence that isn't a letter, hyphen, space, or apostraphe (thanks Copilot)
        # thisTranscription = thisTranscription.replace('--', '').lower() # Replaces '--' with '' (not sure why this wasn't in the re.sub, maybe too complicated?) and turns sentence to lowercase 
        addInterWordSymbol = True
        maxSeqLen = 500

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

    newDataset['textPerTime'] = newDataset['textLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32) # For each trial, divide number of phonemes in sentence by number of time bins for trial to get phonemes per time bin

    # add phoneme versions when requested
    # if output_type == 'both':
    #     newDataset['text2'] = seq_phone
    #     textLens2 = []
    #     for arr in newDataset['text2']:
    #         zeroIdx = np.argwhere(arr == 0)
    #         textLens2.append(zeroIdx[0, 0] if zeroIdx.size else arr.shape[0])
    #     newDataset['textLens2'] = np.array(textLens2)
    #     newDataset['textPerTime2'] = newDataset['textLens2'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)

    return newDataset


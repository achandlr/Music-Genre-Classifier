import os
import torchaudio
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils.dataprocessing import * 
import ast
# import math
import random
# import h5py


class AudioDataset(Dataset):

    def __init__(self, meta_data_path, audio_folder_path, preprocessing_dict = {}, debug = False, datatype = "torch", return_type = tuple, accepted_genres = None):
        self.return_type = return_type
        # genres_jupyter_csv = utils.load('data/fma_metadata/genres.csv')
        tracks_jupyter_csv = load('data/fma_metadata/tracks.csv')
        genres_jupyter = tracks_jupyter_csv.loc[:, ('track', 'genre_top')]

        track_csv = pd.read_csv(meta_data_path+"/tracks.csv", header=1).iloc[1: , :]
        genres_csv = track_csv["genre_top"]
        audio_tensors = [] 
        genres = [] 
        # invalid_genres = set(['nan'])
        if accepted_genres is None:
            accepted_genres = set(['Experimental', 'Pop', 'Folk', 'Electronic',
        'Rock', 'Hip-Hop', 'Instrumental', 'Jazz']) # took out 'International' and "nan"
        not_found_cnt = 0
        torch_audio_read_error_cnt = 0
        small_audio_file_cnt = 0
        bad_genre_cnt = 0
        bad_processing_cnt  = 0
        for subdir, dirs, files in os.walk(audio_folder_path):
            for filename in files:
                # print("Start loop: {}".format(len(genres))) 
                if filename.endswith(('.mp3')):
                    if debug and len(audio_tensors) == 1280:
                        break
                    track_id = eval(filename.rstrip(".mp3").lstrip('0')) 
                    # track_csv_index = track_csv.index[track_csv["Unnamed: 0"] == track_id].tolist()
                    # if not track_csv_index:
                    #     not_found_cnt +=1
                    #     continue
                    # assert len(track_csv_index) == 1
                    # genre = genres_csv.iloc[track_csv_index[0]]
                    # verify genre
                    try:
                        genre = genres_jupyter[track_id]
                    except KeyError:
                        not_found_cnt +=1
                        continue
                    if genre not in accepted_genres:
                        bad_genre_cnt+=1
                        continue
                    # if genre != genre_query_method_2:
                    #     print("manual check")
                    # assert genre == genre_query_method_2
                    # if genre in invalid_genres:
                    #     continue
                    # if math.isnan(genre):
                        # continue
                    #print os.path.join(subdir, file)
                    filepath = subdir + os.sep + filename
                    # print("Loading audio: {}".format(len(genres))) 
                    try:
                        data_waveform, rate_of_sample = torchaudio.load(filepath)
                        preprocessing_dict["orig_freq"] = 16_000 # TODO Alex today
                    except Exception as e:
                        torch_audio_read_error_cnt +=1
                        continue
                    # print("Applying processing: {}".format(len(genres))) 
                    data_waveform = self.apply_preproccess(data_waveform, preprocessing_dict)
                    if data_waveform is None:
                        bad_processing_cnt +=1
                        continue
                    if datatype == "np":
                        data_waveform = data_waveform.detach().numpy()
                    # ignore smaller audio samples (very rarely)
                    # TODO: confirm that replacing 1_300_000 with preprocessing_dict["truncation_len"] does not mess things up
                    # skip sample if not of truncation length
                    if  preprocessing_dict["truncation_len"]!= None and data_waveform.shape[1] < preprocessing_dict["truncation_len"]:
                        small_audio_file_cnt+=1
                        continue
                    audio_tensors.append(data_waveform)
                    genres.append(genre)
        if datatype == "np" and preprocessing_dict["truncation_len"]!= None: #TODO Alex today
            audio_tensors= np.concatenate(audio_tensors) # TODO: rename audio tensors with audio
        elif preprocessing_dict["truncation_len"]!= None: #TODO Alex today
            audio_tensors = torch.stack(audio_tensors)
        genres= np.array(genres)

        self.error_dict = {
        "not_found_cnt" : not_found_cnt,
        "torch_audio_read_error_cnt" : torch_audio_read_error_cnt,
        "small_audio_file_cnt" : small_audio_file_cnt,
        "bad_genre_cnt": bad_genre_cnt,
        "bad_processing_cnt": bad_processing_cnt}

        # genres = genres
        temp = list(zip(audio_tensors, genres))
        random.shuffle(temp)
        audio_tensors, genres = zip(*temp)
        self.audio_tensors = audio_tensors
        self.genres_factorized = pd.factorize(pd.Series(genres))


    def report_data_processing_errors(self):
        return self.error_dict

    def __len__(self):
        assert len(self.audio_tensors) == len(self.genres_factorized[0])
        return len(self.audio_tensors)
    
    def apply_preproccess(self, waveform, proccessing_dict):
        sampling_freq =  proccessing_dict["sampling_freq"]
        orig_freq = proccessing_dict["orig_freq"]
        # if not (not sampling or type(sampling) == dict):
        #     raise ValueError("sampling should either be none or a dictionary but instead is type {}".format(type(sampling)))
        padding_length = proccessing_dict["padding_length"]
        truncation_len = proccessing_dict["truncation_len"]
        convert_one_channel = proccessing_dict["convert_one_channel"]
        if padding_length!=None and truncation_len!=None:
            raise ValueError("Invalid processing parameters. One should not pad and truncate the same sample.")
        
        if sampling_freq != None:
            # orig_freq, new_freq = sampling["orig_freq"], sampling["new_freq"]
            waveform = resample(waveform, orig_freq, sampling_freq)
            # print(waveform.shape)

        # Not necessary for our project
        if padding_length != None:
            raise NotImplementedError()

        if truncation_len != None:
            waveform = truncate_sample(waveform, truncation_len)

        if convert_one_channel != False:
            waveform = convert_to_one_channel(waveform)
        elif waveform.shape[0] !=2:
            # Return None if the waveform is not two channels
            return None
        return waveform

    def __getitem__(self, idx): 
        if self.return_type == tuple:
            return self.audio_tensors[idx], self.genres_factorized[0][idx]
        elif self.return_type == dict:
            return {"input_values": self.audio_tensors[idx].squeeze(), "label": self.genres_factorized[0][idx]} # TODO: check this is probably wrong
        else:
            raise NotImplementedError()


# directly copied from original data location: This is a helper file for loading in the dataset https://github.com/mdeff/fma/blob/master/utils.py
def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks
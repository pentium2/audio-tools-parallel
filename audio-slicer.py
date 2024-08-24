#!/usr/bin/env python3
#
#  Chen Siyu
#  credits: 30% of the code is adapted from 七琦‘s contribution in the Alidenoise project.
#
#
import argparse
import torchfun as tf
import os
import soundfile
import math
import numpy as np
import librosa
import pdb
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser('sound data mixer and sampler, given clean data folderpaths, and given noise wav folderpaths, mix clean and noise wav to generate simulated noisy data. clean/noisy data will be copied into a new folder for further use by pytorch dataset loader')
parser.add_argument('audios',nargs='+',type=str,help='audio dataset foldernames splitted by spaces')

parser.add_argument('--duration',dest='duration',type=int,default=10,help='slice interval in seconds, default 10s')

parser.add_argument('--sampling-rate',dest='sampling_rate',type=int,default=None,help='re-sampling-rate of the generated data, default: Do-not-resmaple')

parser.add_argument('--head-overlap',type=int,default=0,help='overlap at head of each sample in seconds, default 0s; when >0, replicate frames will be used at the start of the first slice clip.')

parser.add_argument('-o','--out-dir',dest='out_dir',type=str,default='output',help='output directory of this program. this dir will be created if not exist, noisy data and clean data will be put into subdirectories respectively.')

parser.add_argument('--dryrun',dest='dryrun',action='store_true',default=False,help='run the process but no actuall change is made.')


def enumerate_wavs_from_folder(folderpath):
    '''given a folder path containing wav files at any level, return the wav file paths as a list.'''
    foldername = os.path.basename(folderpath)
    urls=[]
    for top,d,files in os.walk(folderpath):
        if files:
            for file in files:
                if file.lower().endswith('.wav'):
                    urls.append(os.path.join(top,file))
    return urls

def add_dataset_names_to_each_url(folderpath,urls):
    '''the name of the dataset folder will be used as the name of this dataset.
    each entry inside urls will be replaced with [dataname,url] entry.'''
    dataname = os.path.basename(folderpath)
    urls_with_dataname = [(dataname,url) for url in urls]
    return urls_with_dataname


def batch_get_urls_with_datasetname(folderpath_list):
    '''gather urls with dataset name for a list of data folder to be processed.
    [(datasetname,url)]
    '''
    all_urls_with_dataname = []
    for folder in folderpath_list:
        if folder.endswith(('.wav','.mp3')):
            # this audios input source path is already an audio filename
            urls = [folder]
            folder = os.path.basename(folder).split('.')[0]
        else:
            urls = enumerate_wavs_from_folder(folder)
        urls_with_dataname = add_dataset_names_to_each_url(folder,urls)
        all_urls_with_dataname += urls_with_dataname
    return all_urls_with_dataname
    
def estabilish_output_dir(args):
    from pathlib import Path
    if os.path.exists(args.out_dir):
        tf.warn(f"out put dir {args.out_dir} exists already, forcing output into this folder")
    else:
        tf.info(f"creating {args.out_dir}")
        Path(args.out_dir).mkdir(parents=True,exist_ok=True)

def get_relative_filepath(url,relative_to):
    '''
    Notice: the relative dir contains the relative_to prefix
    '''
    datahome_rel_path_offset = url.find(relative_to)
    rel_path = url[datahome_rel_path_offset:]
    return rel_path

#### audio wave form processing

def read_audio(path, target_fs=None):
    '''read audio from path and resample to target sampling rate'''
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def gen_wavs_for_one_clean(datasetname_urls,sampling_rate,duration,head_overlap,out_dir):
    '''
    datasetname_urls:
        (     (clean_dataname,clean_url) ,  )
    '''
    datasetname,url = datasetname_urls

    relative_url = get_relative_filepath(url,relative_to=datasetname)

    wav,new_sampling_rate = read_audio(url,sampling_rate) # np array
    sampling_rate = new_sampling_rate

    relative_dir,fname,_ = tf.path.get_dir_fname_suffix(relative_url)


    length = len(wav)
    if length == 0:
        tf.warn(f"Zero length audio:{clean_url}")
        return
    # calculate number of slices
    slice_length = int(sampling_rate * duration)
    overlap_length = int(sampling_rate * head_overlap)
    slice_length_non_overlapped = slice_length - overlap_length
    num_slices = math.ceil(length / slice_length_non_overlapped) # the last clip may be shorter than `duration`
    i = 0
    for i in range(num_slices):
        outfpath = os.path.join(out_dir,relative_dir,f'{fname}-{i}.wav')
        outfdir = os.path.dirname(outfpath)
        if not os.path.exists(outfdir):
            os.makedirs(outfdir,exist_ok=True)
        start = i * slice_length_non_overlapped
        end = start+slice_length_non_overlapped
        start = start - overlap_length # may < 0
        frames_to_replicate_and_append = 0 if start>=0 else -start
        start = max(0,start)

        wav_slice = np.concatenate( (wav[:frames_to_replicate_and_append],wav[start:end]),axis=0)

        soundfile.write(file=outfpath,data=wav_slice,samplerate=sampling_rate)

def parallel_gen_wavs_for_one_clean(param):
    '''pack and unfold parameters for gen_wavs_for_one_clean(), in order to achieve parallel execution
    '''
    gen_wavs_for_one_clean(*param)

def main(args):
    #print(args)
    
    datasetname_urls = batch_get_urls_with_datasetname(args.audios)

    numfiles = len(datasetname_urls)

    if numfiles==0:
        tf.warning('no wav files found')
    else:
        tf.warning(f'{numfiles} files found')

    confirm = tf.input_or(f"Data generation plan estabilished:\n \
    input datasets:{','.join(args.audios)} \n \
    input files:{numfiles} wav files \n \
    slice length:{args.duration} sec \n \
    output directory:{args.out_dir}\n\
    Confirm the plan? (y/n)","n") == 'y'

    if not confirm:
        tf.info('exiting')
        return
    
    if args.dryrun:
        tf.info('generation is bypassed')
    else:
        tf.info('start generating')
        estabilish_output_dir(args)

        sampling_rate = args.sampling_rate

        parallel_args = [  [(datasetname,url),sampling_rate,args.duration,args.head_overlap,args.out_dir] for datasetname,url in datasetname_urls ]

        with Pool(processes=multiprocessing.cpu_count()-2) as cpu_pool:
            for _ in tqdm(cpu_pool.imap_unordered(parallel_gen_wavs_for_one_clean,parallel_args),total=len(parallel_args)):
                pass
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)






















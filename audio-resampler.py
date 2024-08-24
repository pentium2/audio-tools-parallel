#!/usr/bin/env python3

import argparse
import os
import soundfile
import wave
import numpy as np
import torchfun as tf
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from scipy.io import wavfile
import scipy.signal as sps
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser('input multiple folders and generate resampled data')
parser.add_argument('input',nargs='+',type=str,help='input dataset foldernames splitted by spaces')

parser.add_argument('--sampling-rate',dest='sampling_rate',type=int,default=16000,help='re-sampling-rate of the generated data, default 16000')

parser.add_argument('-o','--out-dir',dest='out_dir',type=str,default='output',help='output directory of this program. this dir will be created if not exist, noisy data and clean data will be put into subdirectories respectively.')

parser.add_argument('--dryrun',dest='dryrun',action='store_true',default=False,help='run the process but no actuall change is made.')
parser.add_argument('--dryrun-checksr',dest='dryrun_checksr',action='store_true',default=False,help='check sampling rate of existing input dataset')


def enumerate_wavs_from_folder(folderpath):
    '''given a folder path containing wav files at any level, return the wav file paths as a list.
    top contains the folderpath,
    urls do not contain top path/folder path'''
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

def add_output_dir_to_each_url(outputdir,dataset_urls):
    return [(outputdir,dataname,url) for dataname,url in dataset_urls]


def resample_scipy(wavpath,target_sampling_rate):
    (audio, sampling_rate) = soundfile.read(wavpath)
    if sampling_rate == target_sampling_rate:
        return audio
    else:
        number_of_samples = round(len(audio) * float(target_sampling_rate) / sampling_rate)
        resampled_data = sps.resample(audio, number_of_samples)
        return resampled_data

def resample_one_wav(out_data_url,target_sampling_rate):
    out,dataname,url = out_data_url
    origin_fname = os.path.basename(url)
    outpath = os.path.join(out,origin_fname)
    out_dir = os.path.dirname(outpath)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    resampled_data = resample_scipy(url,target_sampling_rate)
    soundfile.write(outpath, resampled_data, target_sampling_rate)

def parallel_process(param):
    out_data_url,target_sampling_rate = param
    resample_one_wav(out_data_url,target_sampling_rate)

def main(args):
    folders=[]
    folder_files=[]
    single_files=[]

    for name in args.input:
        if name.endswith('.wav'):
            single_files.append((args.out_dir,'.',name)) # out,datasetfolder,url
        else:
            folders.append(name)
            urls = enumerate_wavs_from_folder(name)
            dataname_urls = add_dataset_names_to_each_url(name,urls)
            out_dataname_urls = add_output_dir_to_each_url(args.out_dir,dataname_urls)
            folder_files += out_dataname_urls

    all_out_data_urls = folder_files+single_files

    total_num = len(all_out_data_urls)
    datasize = total_num * 0.5 / 1024

    confirm = tf.input_or(f"Data generation plan estabilished:\n \
    Total clean samples:{total_num} \n \
    Expected generation:{total_num},\n \
                        {datasize} GiB \n\
    Confirm the plan? (y/n)","n") == 'y'

    if not confirm:
        logging.info('exiting')
        return

    
    if args.dryrun:
        for outd,dataname,url in all_out_data_urls:
            logging.info(f'{dataname},{url} => {outd}')
        logging.info('generation is bypassed')
    elif args.dryrun_checksr:
        sampling_rates = {}
        print('reading every wav file sr info...')
        for outd,dataname,url in tqdm(all_out_data_urls):
            if os.path.isfile(url):
                file_obj = wave.open(url)
                sr = file_obj.getframerate()
                if sr not in sampling_rates:
                    sampling_rates[sr] = 0
                sampling_rates[sr] = sampling_rates[sr]+1
            else:
                print(f'warn: file not exist {url}')
        for sr in sampling_rates:
            print(f"sampling rate = {sr}, samples = {sampling_rates[sr]}")
    else:
        logging.info('start generating')

        sampling_rate = args.sampling_rate

        parallel_args = [(tup,args.sampling_rate) for tup in all_out_data_urls]

        with Pool(processes=multiprocessing.cpu_count()-2) as cpu_pool:
            for _ in tqdm(cpu_pool.imap_unordered(parallel_process,parallel_args),total=len(parallel_args)):
                pass
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

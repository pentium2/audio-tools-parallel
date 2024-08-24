#!/usr/bin/env python3.6
#
#  Chen Siyu
#  credits: 30% of the code is adapted from 七琦‘s contribution in the Alidenoise project.
#
#
import argparse
import torchfun as tf
import os
import soundfile
import numpy as np
import librosa
import pdb
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from scipy.io import wavfile
import scipy.signal as sps
import logging
from glob import glob

parser = argparse.ArgumentParser('sound data mixer and sampler, given clean data folderpaths, and given noise wav folderpaths, mix clean and noise wav to generate simulated noisy data. clean/noisy data will be copied into a new folder for further use by pytorch dataset loader')
parser.add_argument('clean',nargs='+',type=str,help='clean dataset foldernames splitted by spaces')
parser.add_argument('-n','--noise',dest='noise',nargs='+',type=str,default=None,help='noise dataset foldernames splitted by spaces')

parser.add_argument('--clean-num',dest='clean_num',type=int,default=None,help='number of samples to use in the clean dataset. This limit is not used by default.')
parser.add_argument('--noise-num',dest='noise_num',type=int,default=None,help='number of samples to use in the noise dataset. This limit is not used by default.')

parser.add_argument('--clean-ratio',dest='clean_ratio',type=float,default=1,help='ratio of samples to use in the clean dataset. default to 1.0')
parser.add_argument('--noise-ratio',dest='noise_ratio',type=float,default=1,help='ratio of samples to use in the noise dataset. default to 1.0')

parser.add_argument('--noises-per-clean',dest='noises_per_clean',type=int,default=1,help='how many noises should a clean wav be mixed with.')

parser.add_argument('--snr',dest='snr',type=tf.list_of_int,default=[5],help='signal noise ratio range. please input comma separated float numbers such as 1,2,3,4. \
                            if only one number is given, then single snr is used for all data. \n \
                            if the domain of snr is specified by two numbers: -5,5 e.g. then integer snr will be sampled in this domain\n \
                            if the avaliable snr choices are enumerated using a list like  -5,0,5,15 with no less than 3 snr values, then the snr to be used will be sampled from these choices.')
parser.add_argument('--snr-per-noise',dest='snr_per_noise',type=int,default=1,help='number of snr to use for each noise in the mixing process.')

parser.add_argument('--sampling-rate',dest='sampling_rate',type=int,default=16000,help='re-sampling-rate of the generated data, default 16000')

parser.add_argument('-o','--out-dir',dest='out_dir',type=str,default='output',help='output directory of this program. this dir will be created if not exist, noisy data and clean data will be put into subdirectories respectively.')

parser.add_argument('--get-test',dest='get_test',action='store_true',default=False,help='if this flag is set, the program will output a filename list of samples that can be selected as test data.\
                                                    those files will be moved away to generate new dataset folders')
parser.add_argument('--dryrun',dest='dryrun',action='store_true',default=False,help='run the process but no actuall change is made.')
parser.add_argument('--search-suffix', nargs='*', type=str, default=['wav','flac'], help='searching suffix for output files.')
parser.add_argument('--save-suffix', type=str, default='wav', help='saving suffix for output files.')

CACHE = {}
MAX_SIZE = 100
SUFFIX = 'wav'
SEARCH_SUFFIX = ['wav','flac']

def cache(fname,wav=None):
    print('caching',fname)
    if fname in CACHE:
        return CACHE[fname]
    else:
        if wav is not None:
            if len(CACHE)<MAX_SIZE:
                CACHE[fname]=wav
                return wav
    return None

def resample_scipy(wavpath,input_sampling_rate):
    sampling_rate, data = wavfile.read(wavpath)
    number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
    resample_data = sps.resample(data, number_of_samples)
    data = np.true_divide(resample_data,32768.0)
    soundfile.write(output_file, data, sampling_rate)


def enumerate_wavs_from_folder(folderpath):
    '''given a folder path containing wav files at any level, return the wav file paths as a list.'''
    foldername = os.path.basename(folderpath)
    urls=[]
    for top,d,files in os.walk(folderpath):
        if files:
            for file in files:
                if file.lower().split('.')[-1] in SEARCH_SUFFIX:
                    urls.append(os.path.join(top,file))
    return urls

def add_dataset_names_to_each_url(folderpath,urls):
    '''the name of the dataset folder will be used as the name of this dataset.
    each entry inside urls will be replaced with [dataname,url] entry.'''
    dataname = os.path.basename(folderpath)
    urls_with_dataname = [(dataname,url) for url in urls]
    return urls_with_dataname

def determine_selected_num_of_data(dataset_size,number_of_samples=None,ratio_of_set=1):
    '''calculate the desired number of samples to be used in a dataset'''
    N = dataset_size
    if number_of_samples is not None:
        return min(N,number_of_samples)
    else:
        return min(N,int(ratio_of_set*N))

def determine_snr_sampler(snr):
    '''argument snr can be one nuber, two numbers specifying range, or 3-or-more numbers enumerating possible choices'''
    import numpy as np
    import copy
    snr = copy.deepcopy(snr)
    if len(snr)==1:
        def sampler_iter():
            '''n is the number of samples desired for returning.'''
            while True:
                yield snr[0]
    if len(snr)==2:
        minv,maxv = sorted(snr)
        def sampler_iter():
            sample = np.arange(minv,maxv+1,1)
            while True:
                np.random.shuffle(sample)
                yield from sample.tolist()
    else:
        def sampler_iter():
            while True:
                np.random.shuffle(snr)
                yield from snr

    def sampler(n=1):
        gen = sampler_iter()
        while True:
            yield [next(gen) for i in range(n)]

    return sampler

def batch_get_urls_with_datasetname(folderpath_list):
    '''gather urls with dataset name for a list of data folder to be processed.'''
    all_urls_with_dataname = []
    for folder in folderpath_list:
        urls = enumerate_wavs_from_folder(folder)
        urls_with_dataname = add_dataset_names_to_each_url(folder,urls)
        all_urls_with_dataname += urls_with_dataname
    return all_urls_with_dataname

def get_random_sampled_clean_noise_urls_with_dataname(args):
    '''get selected data entries for both clean and noise dataset'''
    clean_urls_with_dataname = batch_get_urls_with_datasetname(args.clean)
    noise_urls_with_dataname = batch_get_urls_with_datasetname(args.noise)

    clean_num = determine_selected_num_of_data(len(clean_urls_with_dataname),
                                               args.clean_num,
                                               args.clean_ratio)
    noise_num = determine_selected_num_of_data(len(noise_urls_with_dataname),
                                               args.noise_num,
                                               args.noise_ratio)

    # now random select
    np.random.shuffle(clean_urls_with_dataname)
    np.random.shuffle(noise_urls_with_dataname)
    selected_clean_urls_with_dataname = clean_urls_with_dataname[:clean_num] 
    selected_noise_urls_with_dataname = noise_urls_with_dataname[:noise_num]

    return selected_clean_urls_with_dataname,selected_noise_urls_with_dataname

def assign_noises_for_clean_urls(clean_urls_with_dataname,noise_urls_with_dataname,args):
    '''
    output data format:

    List[
            (     (clean_dataname,clean_url) , List[ (noise_dataname,noise_url,[snr1,snr2,...]) ]      ),
            ...
        ]
    '''
    import numpy as np
    NC = args.noises_per_clean
    SN = args.snr_per_noise
    clean_noises_snrs=[]

    snr_sampler = determine_snr_sampler(args.snr)
    snr_gen = snr_sampler(SN)

    def random_noise_list(noise_urls_with_dataname):
        while True:
            np.random.shuffle(noise_urls_with_dataname)
            yield from noise_urls_with_dataname

    noise_iter = random_noise_list(noise_urls_with_dataname)

    for clean_entry in clean_urls_with_dataname:
        noises = []
        np.random.shuffle(noise_urls_with_dataname)
        for i in range(NC):
            noise_dataname,noise_url = next(noise_iter)
            noises.append( (noise_dataname,noise_url,next(snr_gen)) )
        clean_noises_snrs.append((clean_entry,noises))
    return clean_noises_snrs
            
def estabilish_output_dir(args):
    from pathlib import Path
    if os.path.exists(args.out_dir):
        tf.warn(f"out put dir {args.out_dir} exists already, forcing output into this folder")
    else:
        tf.info(f"creating {args.out_dir}")
        Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    clean_dir = os.path.join(args.out_dir,'clean')
    noisy_dir = os.path.join(args.out_dir,'noisy')
    Path(clean_dir).mkdir(parents=True,exist_ok=True)
    Path(noisy_dir).mkdir(parents=True,exist_ok=True)

    return clean_dir,noisy_dir

def move_files_away(urls_with_dataname,out_dir,dryrun=False):
    '''
    urls: [  (datasetname,url)  ]'''
    import shutil
    import os
    import pathlib
    for datasetname, url in urls_with_dataname:
        datahome_rel_path_offset = url.find(datasetname)
        datahome_rel_path = url[datahome_rel_path_offset:]
        out_url = os.path.join(out_dir,datahome_rel_path)
        out_file_dir = os.path.dirname(out_url)
        if dryrun:
            print(url,'->',out_url)
            print('create:',out_file_dir)
        else:
            pathlib.Path(out_file_dir).mkdir(parents=True,exist_ok=True)
            shutil.move(url,out_url)
    return

#### audio wave form processing

def read_audio(path, target_fs=16000):
    '''read audio from path and resample to target sampling rate'''
    try:
        (audio, fs) = soundfile.read(path)
        #fs, audio = wavfile.read(path)
        #audio = np.true_divide(audio,32768.0)
    except Exception as e:
        logging.error(f'error on file:{path}')
        raise e

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        number_of_samples = round(len(audio) * float(target_fs) / fs)
        audio = sps.resample(audio, number_of_samples)
    return audio, target_fs

def get_amplitude_scaling_noise_factor(clean,noise,snr):
    sum_s = np.sum(clean ** 2)
    sum_n = np.sum(noise ** 2)
    scaler = np.sqrt(sum_s/(sum_n * pow(10, (snr/10)) + 1e-08))
    return scaler

def get_mixture_audio(clean_wav,noise_wav,snr):
    noise = noise_wav 
    clean = clean_wav 

    len_speech = len(clean)
    len_noise = len(noise)
    if min(len_speech,len_noise) == 0:
        return None,None 

    if len_noise <= len_speech:
        # Repeat noise to the same length as speech. 
        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
        noise_audio_ex = np.tile(noise, n_repeat)
        noise = noise_audio_ex[0:len_speech]
    # If noise longer than speech then randomly select a segment of noise. 
    else:
        noise_onset = np.random.randint(0, (len_noise - len_speech-1))
        # Truncate noise to the same length as speech.      
        noise = noise[noise_onset:(noise_onset+len_speech)]
    
    scaler = get_amplitude_scaling_noise_factor(clean, noise, snr)
    noise = noise * scaler

    mixture = clean + noise
    max_pluse = np.max(np.abs(mixture))
    if max_pluse == 0:
        raise Exception(f'max pluse val normalization failed: max pluse is zero.')
    alpha = 1. / max_pluse
    mixture = mixture * alpha
    clean = clean * alpha

    assert mixture.shape == clean.shape

    return clean,mixture

def gen_wavs_for_one_clean(one_clean_noises_snrs,sampling_rate,clean_dir,noisy_dir):
    '''
    one_clean_noises_snrs:
        (     (clean_dataname,clean_url) , List[ (noise_dataname,noise_url,[snr1,snr2,...]) ]      )
    '''
    (clean_dataname,clean_url) , noises_with_snrs = one_clean_noises_snrs
    clean_wav,_ = read_audio(clean_url,sampling_rate)
    _,clean_fname,_ = tf.path.get_dir_fname_suffix(clean_url)
    if len(clean_wav) == 0:
        tf.warn(f"Zero length audio:{clean_url}")
        return

    for noise_dataname,noise_url,snrs in noises_with_snrs:
        noise_wav,_ = read_audio(noise_url,sampling_rate)
        _,noise_fname,_ = tf.path.get_dir_fname_suffix(noise_url)
        if len(noise_wav) == 0:
            tf.warn(f"Zero length audio:{noise_url}")
            continue

        for snr in snrs:
            outfname = f"{clean_dataname}+{clean_fname}+{noise_dataname}+{noise_fname}+{snr}.{SUFFIX}"
            
            clean_snrdir = os.path.join(clean_dir,str(snr))
            noisy_snrdir = os.path.join(noisy_dir,str(snr))
            
            if not os.path.exists(clean_snrdir):
                os.makedirs(clean_snrdir,exist_ok=True)
                os.makedirs(noisy_snrdir,exist_ok=True)

            clean_outpath = os.path.join(clean_snrdir,outfname)
            noisy_outpath = os.path.join(noisy_snrdir,outfname)

            if os.path.exists(clean_outpath) and os.path.exists(noisy_outpath):
                # in case that the generation is resumed from last ones
                continue
            else:
                try:
                    clean_final,noisy_final = get_mixture_audio(clean_wav,noise_wav,snr)
                    soundfile.write(file=clean_outpath, data=clean_final, samplerate=sampling_rate)
                    soundfile.write(file=noisy_outpath, data=noisy_final, samplerate=sampling_rate)
                except Exception as e:
                    print(f'Silence may be encountered in clean {clean_fname} and noise {noise_fname}, by passing...') 

def parallel_gen_wavs_for_one_clean(param):
    '''pack and unfold parameters for gen_wavs_for_one_clean(), in order to achieve parallel execution
    '''
    gen_wavs_for_one_clean(*param)

def main(args):
    #print(args)
    #pdb.set_trace()
    global SUFFIX
    SUFFIX = args.save_suffix
    global SEARCH_SUFFIX
    SEARCH_SUFFIX = args.search_suffix

    clean_urls_with_dataname,noise_urls_with_dataname = get_random_sampled_clean_noise_urls_with_dataname(args)
    
    clean_noises_snrs = assign_noises_for_clean_urls(clean_urls_with_dataname,noise_urls_with_dataname,args)

    clean_num = len(clean_noises_snrs)
    noise_num = len(noise_urls_with_dataname)
    noises_per_clean = args.noises_per_clean
    snr_num = args.snr_per_noise
    total_num = clean_num*noises_per_clean*snr_num
    datasize = total_num * 0.5 / 1024

    if args.get_test:
        tf.warning('the get_test flag is set, the following plan is for test data immigration, \
            noise data and clean data are not going to be mixed.')

    confirm = tf.input_or(f"Search for {SEARCH_SUFFIX}, saving to {SUFFIX} \n \
    Data generation plan estabilished:\n \
    Total clean samples:{clean_num} \n \
    Total noise samples:{noise_num} \n \
    Noise variances:{noises_per_clean}\n\
    SNR variances:{snr_num}\n\
    Expected generation:{total_num},\n \
                        {datasize} GiB \n\
    Confirm the plan? (y/n)","n") == 'y'

    if not confirm:
        tf.info('exiting')
        return

    if args.dryrun:
        tf.info('generation is bypassed')
    else:
        tf.info('start generating')
        clean_dir,noisy_dir = estabilish_output_dir(args)

        sampling_rate = args.sampling_rate

        parallel_args = [  [(cleaninfo,noiseinfo),sampling_rate,clean_dir,noisy_dir] for cleaninfo,noiseinfo in clean_noises_snrs ]

        with Pool(processes=multiprocessing.cpu_count()-2) as cpu_pool:
            for _ in tqdm(cpu_pool.imap_unordered(parallel_gen_wavs_for_one_clean,parallel_args),total=len(parallel_args)):
                pass

    if args.get_test:
        move_files_away(clean_urls_with_dataname,args.out_dir,args.dryrun)
        move_files_away(noise_urls_with_dataname,args.out_dir,args.dryrun)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)






















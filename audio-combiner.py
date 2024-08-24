#!/usr/bin/env python3
import sox
from glob import glob
import argparse
import os
import logging
import shutil
import librosa
from tqdm import tqdm

## docs https://github.com/rabitt/pysox

parser  = argparse.ArgumentParser('audio combiner')

parser.add_argument('list',nargs='+',type=str,help='list of file path to be combined into one output, path can contain *')
parser.add_argument('out',type=str,help='output wav name')
parser.add_argument('--sr',type=int,default=16000,help='sample rate, default to 16000')

args = parser.parse_args()

def main(args):
    raw_patterns = []
    for item in args.list:
        if '*' in item:
            item = glob(item,recursive=True)
            raw_patterns = raw_patterns + item
        else:
            raw_patterns.append(item)

    allflist = []
    for ptn in raw_patterns:
        if ptn.lower().endswith('.wav'):
            allflist.append(ptn)
        else:
            allflist = allflist + librosa.util.find_files(ptn)

    logging.info(f"total input files{len(allflist)}")

    outtopdir = os.path.dirname(args.out)
    tmpdir = os.path.join(outtopdir,'tmpwavs')
    tmpdir_fpattern = os.path.join(tmpdir,'*.wav')
    if outtopdir:
        logging.info(f'examine and create output top dir {outtopdir}')
        os.makedirs(outtopdir,exist_ok=True)
    os.makedirs(tmpdir,exist_ok=True)

    tfm = sox.Transformer()
    tfm.convert(samplerate=args.sr,n_channels=1,bitdepth=16)
    for i,f in tqdm(enumerate(allflist),total=len(allflist)):
        outf = os.path.join(tmpdir,f'{i}.wav')
        tfm.build(f,outf)

    tmpfiles = glob(tmpdir_fpattern)

    cbn = sox.Combiner()
    # pitch shift combined audio up 3 semitones
    # cbn.pitch(3.0)
    # convert output to 8000 Hz stereo
    cbn.convert(samplerate=args.sr,n_channels=1,bitdepth=16)
    # create the output file
    cbn.build(
        tmpfiles, args.out, 'concatenate'
    )

    shutil.rmtree(tmpdir,ignore_errors=True)

    return

if __name__ == '__main__':
    main(args)
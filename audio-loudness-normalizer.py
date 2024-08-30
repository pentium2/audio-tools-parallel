import soundfile as sf
import pyloudnorm as lnorm
import os
import argparse
from glob import glob
import multiprocessing as mlp
from tqdm import tqdm

p = argparse.ArgumentParser('''normalize loudness of audio files.
Spotify：-14 LUFS
苹果音乐：-16 LUFS
亚马逊音乐：-9 至 -13 LUFS
Youtube：-13至-15 LUFS
Deezer：-14 至 -16 LUFS
CD:-9 LUFS
声音云：-8 至 -13 LUFS''')
p.add_argument('files',type=str,nargs='+',help='a list of files to process')
p.add_argument('--out','-o',type=str,default=None,help='output dir or filename for a single input')
p.add_argument('--db','-d',type=float,default=-14,help='target loudness in LUFS dB')
p.add_argument('--cores','-c',type=int,default=3,help='number of processes to create for parallel processing.')


def normalize_one_file(fpath,target_loudness_LUFS_dB,output_fpath):
	sig,sr = sf.read(fpath)
	meter = lnorm.Meter(rate=sr)
	loudness_LUFS_dB =  meter.integrated_loudness(sig)
	sig_adjusted = lnorm.normalize.loudness(sig,loudness_LUFS_dB,target_loudness_LUFS_dB)
	sf.write(output_fpath,sig_adjusted,sr)

def parallel_process(params):
	fpath,target_loudness_LUFS_dB,output_fpath = params
	return normalize_one_file(fpath,target_loudness_LUFS_dB,output_fpath)

def main(args):
	files = []
	for f in args.files:
		if os.path.isfile(f):
			files.append(f)
		elif '*' in f:
			sub_files = glob(f,recursive=True)
			files.extend(sub_files)
	out_files = []
	out_name = os.path.basename(args.out)
	top_dir = None
	if '.' in out_name:
		out_files.append(args.out)
		top_dir = os.path.dirname(args.out)
		print(f"output destination is a single file {args.out}")
	else:
		print(f"output destination is treated as a folder {args.out}")
		top_dir = args.out
		for f in files:
			fname = os.path.basename(f)
			target_fpath = os.path.join(args.out,fname)
			out_files.append(target_fpath)

	params_loudness_LUFS_dB = [args.db] * len(files)
	mlp_args = list(zip(files,params_loudness_LUFS_dB,out_files))

	print(f"input {len(files)} files, {len(out_files)} outputs assigned, {len(mlp_args)} tasks prepared for parallel processing.")
	print(f"available cpus={mlp.cpu_count()}, {args.cores} chosen for processing.")
	input(f"enter any key to continue, or ctrl+c to abort.")

	os.makedirs(top_dir,exist_ok=True)
	print(f"confirm to continue, output directory created.")
	with mlp.Pool(processes=args.cores) as cpu_pool:
            for _ in tqdm(cpu_pool.imap_unordered(parallel_process,mlp_args),total=len(mlp_args)):
                pass

if __name__ == "__main__":
	args = p.parse_args()
	main(args)

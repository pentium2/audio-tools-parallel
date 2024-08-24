# audio-tools-parallel
High speed and accurate audio processing toolbox that parallelize most batch audio operations.
- slicer
- resampler
- combiner
- mixer

## Audio Slicer 

multi-thread (posix) accelerated audio slicer.

The output directory will be autimatically created, 
and the source folder structures will be cloned into the output directories.

Sliced wav files will be placed in corresponding places named with integer indexing suffix.

```bash
python3 ./audio-slicer.py folder1 folder2 folder3 \
    --duration 10 \
    --sampling-rate 16000 \
    --out-dir sliced_new_folder
```


## audio resampler
sox and librosa cannot resample correctly because the higher part of the spectrum is usually lost.

we here use scipy resampling algorithm, which is slow but effective.

```bash
python audio-resampler.py --help
```



## Audio combiner
combine multiple wavs into a single wav file. useful when conducting streaming test.

```bash
python3 audio-combiner.py folder1/*.wav 1.wav folder2/* output.wav
```



## Audio mixer

Before executing any command, be sure that you have some folders containing clean wav files, and some folders containing noise wav files.

```
raw_folder/
    tsinghua_dataset/
        clean_folder1/   <- clean_folder1
        noise_folder1/   <- noise_folder1
    clean_folder2/       <- clean_folder2
    noise_folder2/       <- noise_folder2
        babble_noise/
        music_noise/
        stupid_noise/
```

You may have any number of subfolder levels, as all wav files contained will be detected.

This program will mix noisy data, as well as resample clean data so that the sample-rate of clean/noisy speech can match.

### Extract testing/validationg set from the original dataset
```bash
python audio_mixer.py clean_folder1 clean_folder2 --noise noise_folder1 noise_folder2 \
        --get-test \
        --clean-num 100 \
        --noise-num 100 \
        --out-dir validation_set_folder
```
The output folder will be created automatically.


### Mix noise audio with clean speech
```bash
python audio_mixer.py clean_folder1 clean_folder2 --noise noise_folder1 noise_folder2 \
        --noises-per-clean 2 \
        --snr 5 \
        --sampling-rate 16000 \
        --out-dir mixed_data_folder
```
The output folder will be created automatically.

> Notice: After mixing the training set, you shall go to the extracted validation set folder and execute similar command to mix a validation set or testing set. 

### using a range of SNR
sample snr choices from uniform(-5,5)
```bash
python audio_mixer.py clean_folder1 clean_folder2 --noise noise_folder1 noise_folder2 \
        --noises-per-clean 2 \
        --snr [-5,5] \
        --out-dir mixed_data_folder
```

### using a list of SNRs
sample snr choices from the options [-5,-2,0,2,5]
```bash
python audio_mixer.py clean_folder1 clean_folder2 --noise noise_folder1 noise_folder2 \
        --noises-per-clean 2 \
        --snr [-5,-2,0,2,5] \
        --out-dir mixed_data_folder
```

## Dry-run (make plan but don't execute any changes)

```bash
python audio_mixer.py clean_folder1 clean_folder2 --noise noise_folder1 noise_folder2 \
        --out-dir mixed_data_folder \
        --dry-run
```

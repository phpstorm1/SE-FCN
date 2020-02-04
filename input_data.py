"""
Functions for obtaining input feature and label. Variation of some functions are provided for TEDLIUM dataset.
"""

import math
import random
import python_speech_features
import numpy as np
import warnings
import os

from librosa.core import resample
from scipy.io import wavfile

RANDOM_SEED = 3332


def read_wav(wav_path, fs):
    """Read a single-channel wav file from given path. Perform resampling and amp normalization

    :param wav_path: Path where the single-channel wav file is located
    :param fs: Desired sampling rate
    :return: Amp normalized wav at specified sampling rate
    """

    fs_wav, wav = wavfile.read(wav_path)
    wav = wav / np.max(np.abs(wav))

    if fs_wav != fs:
        warnings.warn("Sampling rate of wav file is not ", fs, ". Will be resampled")
        wav = resample(wav, wav, fs)

    return fs_wav, wav / np.max(np.abs(wav))


def get_spectrum(wav, win_len=320, win_shift=160, nDFT=320, win_fun=np.hamming):
    """ Get the spectrum of waveform

    :param wav: single-channel waveform
    :param win_len: window size
    :param win_shift: steps between consecutive frames
    :param nDFT: number of DFT points
    :param win_fun: window function
    :return: magnitude spectrum, phase spectrum, real component of complex spectrum, imaginary component of complex spectrum
    """

    wav_np = np.array(wav).flatten()
    wav_np = np.reshape(wav_np, [len(wav_np)])

    win_len = int(win_len)
    win_shift = int(win_shift)
    nDFT = int(nDFT)

    wav_frame = python_speech_features.sigproc.framesig(sig=wav_np,
                                                        frame_len=win_len,
                                                        frame_step=win_shift,
                                                        winfunc=win_fun)
    wav_fft = np.empty([wav_frame.shape[0], int(win_len / 2 + 1)], dtype=complex)
    for frame in range(wav_frame.shape[0]):
        wav_fft[frame] = np.fft.rfft(a=wav_frame[frame], n=nDFT)
    mag_spectrum = np.abs(wav_fft)
    phase_spectrum = np.arctan2(wav_fft.imag, wav_fft.real)

    return mag_spectrum, phase_spectrum, wav_fft.real, wav_fft.imag


def get_noisy_wav(clean_wav, noise_wav, snr, sensor_noise=None, sensor_snr=15):
    """Generate noisy waveform at given SNR

    :param clean_wav: single-channel waveform of clean speech
    :param noise_wav: single-channel waveform of noise
    :param snr: signal-to-noise ratio of the noisy speech
    :param sensor_noise: white noise produced by sensor, default is None.
    :param sensor_snr: energy ratio of the noisy speech to sensor noise
    :return: noisy speech of certain SNR
    """

    clean_wav = np.array(clean_wav).flatten()
    noise_wav = np.array(noise_wav).flatten()

    clean_wav = clean_wav / np.max(np.abs(clean_wav))
    noise_wav = noise_wav / np.max(np.abs(noise_wav))

    if np.max(np.abs(noise_wav)) == 0:
        raise Exception("the maximum value of the input noise wav is 0")
    if np.max(np.abs(clean_wav)) == 0:
        raise Exception("the maximum value of the input speech wav is 0")

    if len(noise_wav) <= len(clean_wav):
        noise_wav = np.tile(noise_wav, int(math.ceil(len(clean_wav) / len(noise_wav)) + 1))

    rand_starter = random.randint(0, len(noise_wav) - len(clean_wav))
    noise_wav = noise_wav[rand_starter: rand_starter + len(clean_wav)]
    noise_energy = np.sum(noise_wav ** 2, axis=0)

    while noise_energy == 0:
        rand_starter = random.randint(0, len(noise_wav) - len(clean_wav))
        noise_wav = noise_wav[rand_starter: rand_starter + len(clean_wav)]
        noise_energy = np.sum(noise_wav ** 2, axis=0)

    utter_energy = np.sum(clean_wav**2, axis=0)
    if utter_energy == 0:
        raise Exception("Invalid speech")

    if sensor_noise is not None:
        sensor_noise = sensor_noise / np.max(np.abs(sensor_noise))
        if len(sensor_noise) <= len(clean_wav):
            sensor_noise = np.tile(sensor_noise, int(math.ceil(len(clean_wav) / len(sensor_noise)) + 1))
        rand_starter = random.randint(0, len(sensor_noise) - len(clean_wav))
        sensor_noise = sensor_noise[rand_starter: rand_starter + len(clean_wav)]
        mic_noise_energy = np.sum(sensor_noise ** 2, axis=0)
        if mic_noise_energy == 0:
            raise Exception("Mic noise energy is zero")
        energy_ratio = math.sqrt(utter_energy / (mic_noise_energy * (10 ** (sensor_snr / 10))))
        mix = clean_wav + energy_ratio * sensor_noise
        utter_energy = np.sum(mix**2, axis=0)
    else:
        mix = clean_wav

    noise_energy = np.sum(noise_wav**2, axis=0)
    if noise_energy == 0:
        raise Exception("Noise energy is zero")

    energy_ratio = math.sqrt(utter_energy / (noise_energy * (10 ** (snr / 10))))
    mix = mix + noise_wav * energy_ratio

    if np.max(np.abs(mix)) == 0:
        raise Exception("the maximum value of the generated noisy wav is 0")
    if np.isnan(np.sum(mix)):
        raise Exception("NaN in generated noisy wav")
    mix = mix / np.max(np.abs(mix))

    return clean_wav, mix


def get_training_specs(speech_file_list,
                       noise_file_list,
                       snr_list,
                       speech_idx_list,
                       noise_idx_list,
                       snr_idx_list,
                       training_step,
                       batch_size=1,
                       max_wav_length=5,
                       win_len=512,
                       win_shift=256,
                       nDFT=512,
                       context_window_width=13,
                       fs=16e3,
                       sensor_noise_wav=None,
                       sensor_snr=None,
                       sensor_snr_idx_list=None):

    """ Generate spectrum for training the model

    :param speech_file_list: list of speech wav files
    :param noise_file_list: list of noise wab files
    :param snr_list: list of SNR ranges of noisy speech
    :param speech_idx_list: list of speech file index of every training step
    :param noise_idx_list: list of noise file index of every training step
    :param snr_idx_list: list of snr index of every training step
    :param training_step: current training step
    :param batch_size: the number of utterances for current training step
    :param max_wav_length: maximal duration of each utterance in seconds
    :param win_len: size of window
    :param win_shift: steps between consecutive frames
    :param nDFT: number of DFT points
    :param context_window_width: number of input frames
    :param fs: desired sampling frequency
    :param sensor_noise_wav: white noise produced by sensor (microphone, etc.), default is None
    :param sensor_snr: list of energy ratio of the noisy speech to sensor noise
    :param sensor_snr_idx_list: list of sensor snr index of every training step
    :return: real and imaginary component of STFT for training
    """

    stacked_noisy_specs = []
    stacked_utter_specs = []

    batched_noisy_specs = []
    batched_utter_specs = []

    for i in range(batch_size):

        idx = batch_size * training_step + i

        fs_utter, speech_wav = read_wav(speech_file_list[speech_idx_list[idx]], fs)
        fs_noise, noise_wav = read_wav(noise_file_list[noise_idx_list[idx]], fs)

        max_num_samples = int(fs * max_wav_length)
        if len(speech_wav) >= max_num_samples:
            rand_start = random.randint(0, len(speech_wav) - max_num_samples - 1)
            speech_wav = speech_wav[rand_start: rand_start+max_num_samples]

        speech_wav, noisy_wav = get_noisy_wav(speech_wav,
                                              noise_wav,
                                              snr_list[snr_idx_list[idx]],
                                              sensor_noise_wav,
                                              sensor_snr[sensor_snr_idx_list[idx]])

        _, _, noisy_spec_real, noisy_spec_imag = get_spectrum(noisy_wav, win_len, win_shift, nDFT)
        _, _, utter_spec_real, utter_spec_imag = get_spectrum(speech_wav, win_len, win_shift, nDFT)

        noisy_specs = np.stack((noisy_spec_real, noisy_spec_imag), axis=2)
        utter_specs = np.stack((utter_spec_real, utter_spec_imag), axis=2)

        stacked_noisy_specs.append(noisy_specs[:-context_window_width+1, :, :])
        stacked_utter_specs.append(utter_specs[:-context_window_width+1, :, :])

        size_one_batched_noisy_specs = batch_spec(noisy_specs, context_window_width)
        size_one_batched_utter_specs = batch_spec(utter_specs, context_window_width)

        batched_noisy_specs.append(size_one_batched_noisy_specs)
        batched_utter_specs.append(size_one_batched_utter_specs)

    batched_noisy_specs = np.vstack(np.array(batched_noisy_specs))
    batched_utter_specs = np.vstack(np.array(batched_utter_specs))

    stacked_noisy_specs = np.vstack(np.array(stacked_noisy_specs))
    stacked_utter_specs = np.vstack(np.array(stacked_utter_specs))

    return batched_utter_specs, batched_noisy_specs, stacked_utter_specs, stacked_noisy_specs


def batch_spec(spec, context_window_width):
    """ Generate batched spectrum

    :param spec: spectrum of speech
    :param context_window_width: number of input frames
    :return: batched spectrum of speech
    """

    num_frames = spec.shape[0]
    frame_len = spec.shape[1]
    num_batch = num_frames - context_window_width + 1
    batched = np.zeros([num_batch, context_window_width, frame_len, 2])
    for i in range(num_batch):
        batched[i, :, :, :] = spec[i:i + context_window_width, :, :]
    return batched


def get_testing_specs(utter_wav,
                      noisy_wav,
                      win_len=512,
                      win_shift=256,
                      nDFT=512,
                      context_window_width=13):
    """ Get spectrum of clean and noisy speech for testing

    :param utter_wav: single-channel clean speech
    :param noisy_wav: single-channel noisy speech
    :param win_len: window size
    :param win_shift: steps between consecutive frames
    :param nDFT: number of DFT points
    :param context_window_width: number of input frames
    :return: input and target complex-valued spectrum
    """

    _, _, noisy_spec_real, noisy_spec_imag = get_spectrum(noisy_wav, win_len, win_shift, nDFT)
    _, _, utter_spec_real, utter_spec_imag = get_spectrum(utter_wav, win_len, win_shift, nDFT)

    noisy_specs = np.stack((noisy_spec_real, noisy_spec_imag), axis=2)
    utter_specs = np.stack((utter_spec_real, utter_spec_imag), axis=2)

    batched_noisy_specs = batch_spec(noisy_specs, context_window_width)
    batched_utter_specs = batch_spec(utter_specs, context_window_width)

    windowed_utter_specs = np.array(utter_specs[:-context_window_width+1, :, :])
    windowed_noisy_specs = np.array(noisy_specs[:-context_window_width+1, :, :])

    return batched_utter_specs, batched_noisy_specs, windowed_utter_specs, windowed_noisy_specs


def get_seg_testing_specs(mix_wav, wav_length_per_seg, seg_idx, win_len, win_shift, nDFT, time_steps, fs):
    segment = int(math.ceil(len(mix_wav) / (wav_length_per_seg * fs)))
    seg_num_frames = int(math.ceil((wav_length_per_seg * fs - win_len) / win_shift) + 1)

    mix_wav = mix_wav / np.max(np.abs(mix_wav))
    _, _, mix_spectrogram_real, mix_spectrogram_imag = get_spectrum(mix_wav, win_len, win_shift, nDFT)

    if seg_idx != segment - 1 and seg_idx != 0:
        start_idx = seg_idx * seg_num_frames - time_steps + 1
        end_idx = (seg_idx + 1) * seg_num_frames
        seg_mix_spectrogram_real = mix_spectrogram_real[start_idx:end_idx, :]
        seg_mix_spectrogram_imag = mix_spectrogram_imag[start_idx:end_idx, :]
    elif seg_idx == 0:
        start_idx = 0
        end_idx = seg_num_frames
        seg_mix_spectrogram_real = mix_spectrogram_real[start_idx:end_idx, :]
        seg_mix_spectrogram_imag = mix_spectrogram_imag[start_idx:end_idx, :]
        seg_mix_spectrogram_real = np.concatenate((seg_mix_spectrogram_real[:time_steps, :], seg_mix_spectrogram_real), axis=0)
        seg_mix_spectrogram_imag = np.concatenate((seg_mix_spectrogram_imag[:time_steps, :], seg_mix_spectrogram_imag), axis=0)
    else:
        start_idx = seg_idx * seg_num_frames - time_steps + 1
        seg_mix_spectrogram_real = mix_spectrogram_real[start_idx:, :]
        seg_mix_spectrogram_imag = mix_spectrogram_imag[start_idx:, :]

    number_of_frames = np.shape(seg_mix_spectrogram_real)[0]
    if number_of_frames <= time_steps:
        warnings.warn("Skip the last segment as the number of frames is smaller than context window length", Warning)

    noisy_specs = np.stack((seg_mix_spectrogram_real, seg_mix_spectrogram_imag), axis=2)
    noisy_specs = batch_spec(noisy_specs, time_steps)

    return noisy_specs


def get_training_wav_tedlium(speech_file_list,
                             noise_file_list,
                             snr_list,
                             speech_idx_list,
                             noise_idx_list,
                             snr_idx_list,
                             stm_path,
                             utter_pos_idx_list,
                             training_step,
                             utter_percentage=0.8,
                             fs=16000,
                             sensor_noise_wav=None,
                             sensor_snr=None,
                             sensor_snr_idx_list=None
                             ):

    """ FOR TEDLIUM DATASET ONLY: Get one utterance from tedlium dataset, generate its noisy form at given snr

    :param speech_file_list: list of ted-talk wav files
    :param noise_file_list: list of noise wav files
    :param snr_list: list of snr ranges for generating noisy speech
    :param speech_idx_list: list of speech file index of every training step
    :param noise_idx_list: list of noise file index of every training step
    :param snr_idx_list: list of snr index of every training step
    :param stm_path: path containing '.stm' files of the ted-lium dataset
    :param utter_pos_idx_list: list of index of utterance position for every training step
    :param training_step: current training step
    :param utter_percentage: the portion of speech to the whole utterance (speech + silence)
    :param fs: sampling frequency
    :param sensor_noise_wav: white noise produced by sensor (microphone, etc.), default is None
    :param sensor_snr: list of energy ratio of the noisy speech to sensor noise
    :param sensor_snr_idx_list: list of sensor snr index of every training step
    :return: segmented extended clean utterance, segmented noisy utterance
    """

    _, speech_wav = read_wav(speech_file_list[speech_idx_list[training_step]], fs)
    _, noise_wav = read_wav(noise_file_list[noise_idx_list[training_step]], fs)

    stm_file = os.path.basename(speech_file_list[speech_idx_list[training_step]]).split(".wav")[0] + ".stm"
    stm_file = os.path.join(stm_path, stm_file)
    utter_pos = get_utter_pos(stm_file, fs)
    pos_idx = math.floor(utter_pos.shape[0] * utter_pos_idx_list[training_step])

    padded_speech, noisy_wav, len_utter, padding_len = get_noisy_wav_tedlium(speech_wav,
                                                                             noise_wav,
                                                                             utter_pos,
                                                                             pos_idx,
                                                                             utter_percentage,
                                                                             snr_list[snr_idx_list[training_step]],
                                                                             fs,
                                                                             sensor_noise_wav,
                                                                             sensor_snr[sensor_snr_idx_list[training_step]])

    return padded_speech, noisy_wav


def get_noisy_wav_tedlium(clean_wav, noise_wav, utter_pos, pos_index, utter_percentage, snr, fs=16000, sensor_noise=None, sensor_snr=15):
    """ FOR TEDLIUM DATASET ONLY: Generate noisy speech from the clean one and noise.

    :param clean_wav: a complete ted talk from tedlium dataset
    :param noise_wav: single-channel noise wav
    :param utter_pos: the position of each utterance of the talk
    :param pos_index: index of the desired utterance
    :param utter_percentage: the portion of speech to the whole utterance (speech + silence)
    :param fs: sampling frequency
    :param snr: signal-to-noise ratio of the noisy speech
    :param sensor_noise: white noise produced by sensor (microphone, etc.), default is None
    :param sensor_snr: energy ratio of the noisy speech to sensor noise
    :return: extended clean utterance and the noisy utterance
    """

    clean_wav = np.array(clean_wav).flatten()
    noise_wav = np.array(noise_wav).flatten()

    clean_wav = clean_wav / np.max(np.abs(clean_wav))
    noise_wav = noise_wav / np.max(np.abs(noise_wav))

    if np.max(np.abs(noise_wav)) == 0:
        raise Exception("the maximum value of the input noise wav is 0")
    if np.max(np.abs(clean_wav)) == 0:
        raise Exception("the maximum value of the input speech wav is 0")

    start_position = int(utter_pos[pos_index][0])
    end_position = int(utter_pos[pos_index][1])

    utter = clean_wav[start_position:end_position]
    desired_len = int(len(utter) / utter_percentage)
    # desired_len = int(math.ceil(desired_len / fs) * fs)
    padding_len = int((desired_len - len(utter)) / 2)

    zero_padding = np.zeros([padding_len, 1]).flatten()
    padded_utter = np.concatenate((zero_padding, utter, zero_padding), axis=0)

    if len(noise_wav) <= len(padded_utter):
        noise_wav = np.tile(noise_wav, int(math.ceil(desired_len / len(noise_wav)) + 1))

    rand_starter = random.randint(0, len(noise_wav) - len(padded_utter))
    noise_wav = noise_wav[rand_starter: rand_starter + len(padded_utter)]

    ext_utter_energy = np.sum(padded_utter ** 2, axis=0)
    if ext_utter_energy == 0:
        raise Exception("Invalid va position")

    if sensor_noise is not None:
        sensor_noise = sensor_noise / np.max(np.abs(sensor_noise))
        if len(sensor_noise) <= len(padded_utter):
            sensor_noise = np.tile(sensor_noise, int(math.ceil(desired_len / len(sensor_noise)) + 1))
        rand_starter = random.randint(0, len(sensor_noise) - len(padded_utter))
        sensor_noise = sensor_noise[rand_starter: rand_starter + len(padded_utter)]
        mic_noise_energy = np.sum(sensor_noise ** 2, axis=0)
        if mic_noise_energy == 0:
            raise Exception("Mic noise energy is zero")
        energy_ratio = math.sqrt(ext_utter_energy / (mic_noise_energy * (10 ** (sensor_snr / 10))))
        mix = padded_utter + energy_ratio * sensor_noise
        ext_utter_energy = np.sum(mix ** 2, axis=0)
    else:
        mix = padded_utter

    noise_energy = np.sum(noise_wav ** 2, axis=0)
    if noise_energy == 0:
        raise Exception("Noise energy is zero")

    energy_ratio = math.sqrt(ext_utter_energy / (noise_energy * (10 ** (snr / 10))))
    mix = mix + noise_wav * energy_ratio

    if np.max(np.abs(mix)) == 0:
        raise Exception("the maximum value of the generated noisy wav is 0")
    if np.isnan(np.sum(mix)):
        raise Exception("NaN in generated noisy wav")
    mix = mix / np.max(np.abs(mix))

    return padded_utter, mix, len(utter), padding_len


def get_utter_pos(stm_filename, fs=16000):
    """ Get the position in time of each utterance

    :param stm_filename: the corresponding .stm file name of the tedlium wav file.
    :param fs: sampling frequency
    :return: matrix of size [number of utterances, 2] containing the start and end of each utterance
    """

    lines = open(stm_filename, encoding="utf8").read().splitlines()
    utter_pos = np.zeros([len(lines), 2])
    for index, line in enumerate(lines):
        start_time = float(line.split(' ')[3])
        end_time = float(line.split(' ')[4])
        utter_pos[index, 0] = int(start_time*fs)
        utter_pos[index, 1] = int(end_time*fs)
    return utter_pos


def get_seg_specs(utter_wav, mix_wav, wav_length_per_seg, seg_idx, win_len, win_shift, nDFT, context_window_width, fs):
    """ Segment wav of excessive duration and get the corresponding spectrum.

    :param utter_wav: wav of single-channel clean speech
    :param mix_wav: wav of single-channel noisy speech
    :param wav_length_per_seg: duration of each segment
    :param seg_idx: the index of output segment
    :param win_len: window size
    :param win_shift: steps between consecutive frames
    :param nDFT: number of DFT points
    :param context_window_width: number of input frames
    :param fs: sampling frequency
    :return: complex-valued spectrum of the segmented wav of index seg_idx.
    """

    segment = int(math.ceil(len(mix_wav) / (wav_length_per_seg * fs)))
    seg_num_frames = int(math.ceil((wav_length_per_seg * fs - win_len) / win_shift) + 1)
    _, _, noisy_spec_real, noisy_spec_imag = get_spectrum(mix_wav, win_len, win_shift, nDFT)
    _, _, utter_spec_real, utter_spect_imag = get_spectrum(utter_wav, win_len, win_shift, nDFT)
    if seg_idx != segment - 1 and seg_idx != 0:
        start_idx = seg_idx * seg_num_frames - context_window_width + 1
        end_idx = (seg_idx + 1) * seg_num_frames
        seg_noisy_spec_real = noisy_spec_real[start_idx:end_idx, :]
        seg_noisy_spec_imag = noisy_spec_imag[start_idx:end_idx, :]
        seg_utter_spec_real = utter_spec_real[start_idx:end_idx, :]
        seg_utter_spec_imag = utter_spect_imag[start_idx:end_idx, :]
    elif seg_idx == 0:
        start_idx = 0
        end_idx = seg_num_frames
        seg_noisy_spec_real = noisy_spec_real[start_idx:end_idx, :]
        seg_noisy_spec_imag = noisy_spec_imag[start_idx:end_idx, :]
        seg_utter_spec_real = utter_spec_real[start_idx:end_idx, :]
        seg_utter_spec_imag = utter_spect_imag[start_idx:end_idx, :]
    else:
        start_idx = seg_idx * seg_num_frames - context_window_width + 1
        seg_noisy_spec_real = noisy_spec_real[start_idx:, :]
        seg_noisy_spec_imag = noisy_spec_imag[start_idx:, :]
        seg_utter_spec_real = utter_spec_real[start_idx:, :]
        seg_utter_spec_imag = utter_spect_imag[start_idx:, :]

    number_of_frames = np.shape(seg_noisy_spec_real)[0]
    if number_of_frames <= context_window_width:
        warnings.warn("Skip the last segment as the number of frames is smaller than context window length", Warning)

    complex_spec = np.stack((seg_noisy_spec_real, seg_noisy_spec_imag), axis=2)
    noisy_complex_spec = batch_spec(complex_spec, context_window_width)

    ground_truth_spec_real = seg_utter_spec_real[context_window_width-1:, :]
    ground_truth_spec_imag = seg_utter_spec_imag[context_window_width-1:, :]
    ground_truth_complex_spec = np.stack((ground_truth_spec_real, ground_truth_spec_imag), axis=2)

    return noisy_complex_spec, ground_truth_complex_spec

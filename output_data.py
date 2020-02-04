import numpy as np
import python_speech_features as pysp
import scipy.io.wavfile
import warnings


def rec_wav(mag_spec, phase_spec=None, spec_imag=None, win_len=320, win_shift=160, nDFT=320, win_fun=np.hamming):
    """ Reconstruct single-channel time-domain waveform from either magnitude and phase spectrum, or real and imaginary component of STFT

    :param mag_spec: spectral of real STFT component if phase_spec is None, magnitude spectrum otherwise
    :param phase_spec: phase spectrum
    :param spec_imag: imaginary STFT component
    :param win_len: window size
    :param win_shift: steps between consecutive frames
    :param nDFT: number of DFT points
    :param win_fun: window function
    :return: single-channel waveform
    """

    mag = np.array(mag_spec)

    if phase_spec is not None:
        phase = np.array(phase_spec)
        if mag.shape != phase.shape:
            raise Exception("The shape of mag_spectrum and phase_spectrum doesn't match")
        rec_fft = np.multiply(mag, np.exp(1j*phase))
    elif spec_imag is not None:
        fft_imag = np.array(spec_imag)
        if mag.shape != fft_imag.shape:
            raise Exception("The shape of mag_spectrum and additional_mag_spectrum doesn't match")
        rec_fft = mag_spec + 1.0j * fft_imag
    else:
        raise Exception("Invalid input: both phase_spectrum and additional_mag_spectrym are missing")

    wav_ifft = np.fft.irfft(a=rec_fft, n=nDFT, axis=1)
    wav_ifft = wav_ifft[:, :win_len]

    wav_deframe = pysp.sigproc.deframesig(frames=wav_ifft,
                                          siglen=0,
                                          frame_len=win_len,
                                          frame_step=win_shift,
                                          winfunc=win_fun
                                          )

    # set first frame and last frame to zeros to get rid of the impulse which seems to be caused by STFT
    wav_deframe[0:win_len] = 0
    wav_deframe[-win_len:] = 0

    nan_idx = np.argwhere(np.isnan(wav_deframe))
    if len(nan_idx):
        warnings.warn("Warning: NaN in wav_deframe")
        wav_deframe[nan_idx] = 0
    if np.max(abs(wav_deframe)) == 0:
        warnings.warn("Warning: zeros array for wav_deframe")
    else:
        wav_deframe = wav_deframe / np.max(abs(wav_deframe))
    return wav_deframe


def save_wav_file(save_path, wav_data, sample_rate):
    """Saves audio data to .wav audio file.

    Args:
      save_path: Path to save the file to.
      wav_data: Array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    wav_data = wav_data * (2**15-1)
    wav_data = wav_data.astype(np.int16, order='C')
    scipy.io.wavfile.write(save_path, sample_rate, wav_data)


def rec_batch(input_spec, output_spec, batch_idx):
    rec_spec = np.zeros(input_spec.shape)
    output_batch_idx = 0
    for idx, elem in enumerate(batch_idx):
        if not elem:
            rec_spec[idx, :, :, :] = output_spec[output_batch_idx, :, :, :]
            output_batch_idx += 1
        else:
            rec_spec[idx, :, :, :] = input_spec[idx, :, :, :]
    return rec_spec


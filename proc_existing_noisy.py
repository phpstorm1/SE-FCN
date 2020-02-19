import os.path
import sys
import json

from functools import reduce
from scipy.io import wavfile

import tensorflow as tf
import numpy as np
import math
import fnmatch

import model
import input_data
import output_data

def main(_):

    # import config
    json_dir = './config.json'
    with open(json_dir) as config_json:
        config = json.load(config_json)

    # define noisy specs
    input_specs = tf.placeholder(tf.float32, shape=[None, config['context_window_width'], 257, 2], name='specs')

    # create SE-FCN
    with tf.variable_scope('SEFCN'):
        model_out = model.se_fcn(input_specs, config['nDFT'], config['context_window_width'])
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SEFCN')

    print('-' * 80)
    print('SE-FCN vars')
    nparams = 0
    for v in model_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    sess = tf.InteractiveSession()
    model_path = os.path.join(config['model_dir'], config['param_file'])

    # load model parameters from checkpoint
    model.load_variables_from_checkpoint(sess, model_path)

    for root, dirs, files in os.walk(config['recording_dir']):
        for basename in files:

            if not fnmatch.fnmatch(basename, '*.wav'):
                continue

            subdir = os.path.basename(os.path.normpath(root))
            filename = os.path.join(root, basename)
            fs, mix_wav = wavfile.read(filename)
            mix_wav = mix_wav / (2**15-1)
            max_amp = np.max(np.abs(mix_wav))

            segment = int(math.ceil(len(mix_wav) / (config['seg_recording_length'] * fs)))
            for segment_idx in range(segment):

                testing_specs = input_data.get_seg_testing_specs(
                    mix_wav=mix_wav,
                    fs=fs,
                    wav_length_per_seg=config['seg_recording_length'],
                    seg_idx=segment_idx,
                    win_len=config['win_len'],
                    win_shift=config['win_shift'],
                    nDFT=config['nDFT'],
                    context_window=config['context_window_width'])

                seg_specs = sess.run([model_out], feed_dict={input_specs: testing_specs})
                print("processing file: " + filename, " "*5,
                      "seg:", "{}/{}".format(segment_idx+1, segment), " "*5,
                      "proc num batch:", testing_specs.shape[0])

                seg_specs = np.vstack(seg_specs)
                seg_specs_real = seg_specs[:, :, 0]
                seg_specs_imag = seg_specs[:, :, 1]

                if segment_idx == 0:
                    rec_test_out_real = seg_specs_real
                    rec_test_out_imag = seg_specs_imag
                else:
                    rec_test_out_real = np.concatenate((rec_test_out_real, seg_specs_real), axis=0)
                    rec_test_out_imag = np.concatenate((rec_test_out_imag, seg_specs_imag), axis=0)

            rec_wav = output_data.rec_wav(mag_spec=rec_test_out_real,
                                          spec_imag=rec_test_out_imag,
                                          win_len=config['win_len'],
                                          win_shift=config['win_shift'],
                                          nDFT=config['nDFT'])
            rec_wav = rec_wav * max_amp
            save_path = os.path.join(config['save_processed_recordings_dir'], subdir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            comp_save_path = os.path.join(save_path, basename)
            output_data.save_wav_file(comp_save_path, rec_wav, fs)

    np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

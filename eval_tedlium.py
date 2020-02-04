import os.path
import sys
import glob
import json

from functools import reduce

import tensorflow as tf
import numpy as np
import math

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

    # define clean specs
    target_specs = tf.placeholder(tf.float32, shape=[None, 257, 2], name='ground_truth')

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

    # define loss and the optimizer
    mse = tf.losses.mean_squared_error(target_specs, model_out)

    sess = tf.InteractiveSession()
    model_path = os.path.join(config['model_dir'], config['param_file'])

    # load model parameters from checkpoint
    model.load_variables_from_checkpoint(sess, model_path)

    # run the test & save the test results

    tf.logging.set_verbosity(tf.logging.ERROR)

    testing_file_list = glob.glob(os.path.join(config['test_tedlium_wav_dir'], "*.wav"))
    print('testing set size: ', len(testing_file_list))
    test_snr = config['test_snr']

    for file_idx, testing_file in enumerate(testing_file_list):

        _, clean_wav = input_data.read_wav(testing_file, config['sampling_rate'])
        stm_path = os.path.join(config['stm_path'], os.path.basename(testing_file).split(".wav")[0] + '.stm')
        utter_pos = input_data.get_utter_pos(stm_path, config['sampling_rate'])

        for noise_idx in range(len(config['test_noise'])):

            noise_wav_path = os.path.join(config['test_noise_path'], config['test_noise'][noise_idx] + '.wav')
            _, noise_wav = input_data.read_wav(noise_wav_path, config['sampling_rate'])

            for snr_idx in range(len(test_snr)):

                for utter_index in range(config['how_many_testing_utter']):
                    utter_wav, noisy_wav, _, _ = input_data.get_noisy_wav_tedlium(clean_wav=clean_wav,
                                                                                  noise_wav=noise_wav,
                                                                                  utter_pos=utter_pos,
                                                                                  pos_index=utter_index,
                                                                                  snr=test_snr[snr_idx],
                                                                                  utter_percentage=config['speech_percentage'])

                    segment = int(math.floor(len(noisy_wav) / (config['wav_length_per_batch'] * config['sampling_rate'])))

                    for segment_idx in range(segment):
                        noisy_specs, clean_specs = input_data.get_seg_specs(mix_wav=noisy_wav,
                                                                            utter_wav=utter_wav,
                                                                            wav_length_per_seg=config['wav_length_per_batch'],
                                                                            seg_idx=segment_idx,
                                                                            win_len=config['win_len'],
                                                                            win_shift=config['win_shift'],
                                                                            context_window_width=config['context_window_width'],
                                                                            fs=config['sampling_rate'],
                                                                            nDFT=config['nDFT'])

                        seg_specs, seg_mse = sess.run([model_out, mse],
                                                      feed_dict={input_specs: noisy_specs, target_specs: clean_specs})

                        print("processing file: " + testing_file, " " * 5,
                              "seg:", "{}/{}".format(segment_idx, segment), " " * 5,
                              "proc num batch:", input_specs.shape[0], " " * 5,
                              "seg mse:", format(seg_specs, '.5f'))

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

                    save_path = os.path.join(config['save_testing_results_dir'],
                                             'test',
                                             str(config['noise_type'][noise_idx]),
                                             str(test_snr[snr_idx]))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    comp_save_path = os.path.join(save_path, os.path.basename(testing_file).split(".wav")[0] + '_U' + str(utter_index) + '.wav')
                    output_data.save_wav_file(comp_save_path, rec_wav, config['sampling_rate'])

                    save_path = os.path.join(config['save_testing_results_dir'],
                                             'mix',
                                             str(config['noise_type'][noise_idx]),
                                             str(test_snr[snr_idx]))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    comp_save_path = os.path.join(save_path, os.path.basename(testing_file).split(".wav")[0] + '_U' + str(utter_index) + '.wav')
                    output_data.save_wav_file(comp_save_path, noisy_wav, config['sampling_rate'])

                    save_path = os.path.join(config['save_testing_results_dir'],
                                             'clean',
                                             str(config['noise_type'][noise_idx]),
                                             str(test_snr[snr_idx]))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    comp_save_path = os.path.join(save_path, os.path.basename(testing_file).split(".wav")[0] + '_U' + str(utter_index) + '.wav')
                    output_data.save_wav_file(comp_save_path, utter_wav, config['sampling_rate'])

    np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

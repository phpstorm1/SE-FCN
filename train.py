import os.path
import sys
import glob
import json
import random

import tensorflow as tf
import numpy as np

import model
import input_data
import output_data

from functools import reduce


def main(_):

    # random seed
    RANDOM_SEED = 3233

    # import config
    json_dir = './config.json'
    with open(json_dir) as config_json:
        config = json.load(config_json)

    # define noisy specs
    input_specs = tf.placeholder(tf.float32, shape=[None, config['context_window_width'], 257, 2], name='specs')

    # define clean specs
    train_target = tf.placeholder(tf.float32, shape=[None, 257, 2], name='ground_truth')

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
    mse = tf.losses.mean_squared_error(train_target, model_out)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope('train'), tf.control_dependencies(extra_update_ops):
        adam = tf.train.AdamOptimizer(config['Adam_learn_rate'])
        train_op = adam.minimize(mse, global_step=global_step, var_list=model_vars)

    # make summaries
    tf.summary.scalar('mse', mse)

    # train the model
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config['summaries_dir'], sess.graph)
    tf.train.write_graph(sess.graph_def, config['train_dir'], 'model.pbtxt')

    tf.global_variables_initializer().run()
    init_local_variable = tf.local_variables_initializer()
    init_local_variable.run()

    start_step = 0
    if config['start_checkpoint']:
        model.load_variables_from_checkpoint(sess, config['start_checkpoint'])
        start_step = global_step.eval(session=sess)
    print('Training from step:', start_step)

    tf.logging.set_verbosity(tf.logging.ERROR)

    snr = range(config['snr_range'][0], config['snr_range'][1] + 1)
    sensor_snr = range(config['sensor_snr_range'][0], config['sensor_snr_range'][1] + 1)

    speech_file_list = glob.glob(os.path.join(config['training_data_dir'], "**", "*.wav"), recursive=True)
    noise_file_list = glob.glob(os.path.join(config['noise_dir'], "**", "*.wav"), recursive=True)

    if not len(speech_file_list):
        Exception("No wav files found at " + config['training_data_dir'])
    if not len(noise_file_list):
        Exception("No wav files found at " + config['noise_dir'])

    # get amp normalized sensor noise data
    if len(config['sensor_noise_path']):
        fs_sensor, sensor_wav = input_data.read_wav(config['sensor_noise_path'], config['sampling_rate'])
    else:
        sensor_wav = None

    print("Number of training speech wav files: ", len(speech_file_list))
    print("Number of training noise wav files: ", len(noise_file_list))

    how_many_training_steps = config['how_many_training_steps']

    random.seed(RANDOM_SEED)
    rand_noise_file_idx_list = [random.randint(0, len(noise_file_list)-1)
                                for i in range(int(how_many_training_steps + 1) * config['batch_size'])]

    random.seed(RANDOM_SEED)
    rand_speech_file_idx_list = [random.randint(0, len(speech_file_list)-1)
                                 for i in range(int(how_many_training_steps + 1) * config['batch_size'])]

    random.seed(RANDOM_SEED)
    snr_idx_list = [random.randint(0, len(snr)-1) for i in range(int(how_many_training_steps + 1) * config['batch_size'])]

    random.seed(RANDOM_SEED)
    sensor_snr_idx_list = [random.randint(0, len(snr)-1) for i in range(int(how_many_training_steps + 1) * config['batch_size'])]

    for training_step in range(start_step+1, int(how_many_training_steps + 1)):

        # get training data
        _, batch_noisy_specs, speech_specs, _ = input_data.get_training_specs(speech_file_list,
                                                                              noise_file_list,
                                                                              snr,
                                                                              rand_speech_file_idx_list,
                                                                              rand_noise_file_idx_list,
                                                                              snr_idx_list,
                                                                              sensor_noise_wav=sensor_wav,
                                                                              sensor_snr=sensor_snr,
                                                                              sensor_snr_idx_list=sensor_snr_idx_list,
                                                                              training_step=training_step,
                                                                              batch_size=config['batch_size'],
                                                                              context_window_width=config['context_window_width'])

        # train the model
        _, training_summary, train_mse = sess.run([train_op, merged_summaries, mse],
                                                  feed_dict={input_specs: batch_noisy_specs, train_target: speech_specs})

        print("training step:", training_step, " "*10, "mse:", format(train_mse, '.5f'))
        train_writer.add_summary(training_summary, training_step)

        # Save the model checkpoint periodically.
        if training_step % config['save_checkpoint_steps'] == 0 or training_step == how_many_training_steps:
            checkpoint_path = os.path.join(config['train_dir'], 'sefcn.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    # run the test & save the test results
    # find testing files
    testing_file_list = glob.glob(os.path.join(config['testing_data_dir'], "*.wav"))
    print('testing set size: ', len(testing_file_list))

    test_snr = config['test_snr']
    overall_testing_mse = np.zeros([len(snr), len(config['test_noise'])])

    for file_idx, testing_file in enumerate(testing_file_list):

        _, clean_wav = input_data.read_wav(testing_file, config['sampling_rate'])

        for noise_idx in range(len(config['test_noise'])):

            noise_wav_path = os.path.join(config['test_noise_path'], config['test_noise'][noise_idx] + '.wav')
            _, noise_wav = input_data.read_wav(noise_wav_path, config['sampling_rate'])

            for snr_idx in range(len(test_snr)):

                utter_wav, noisy_wav = input_data.get_noisy_wav(clean_wav=clean_wav,
                                                                noise_wav=noise_wav,
                                                                snr=test_snr[snr_idx])

                _, batched_noisy_specs, speech_specs, _ = input_data.get_testing_specs(utter_wav,
                                                                                       noisy_wav,
                                                                                       context_window_width=config['context_window_width'])

                estimate_specs, test_mse = sess.run([model_out, mse],
                                                    feed_dict={input_specs: batched_noisy_specs, train_target: speech_specs})

                rec_wav = output_data.rec_wav(mag_spec=estimate_specs[:, :, 0],
                                              spec_imag=estimate_specs[:, :, 1],
                                              win_len=config['win_len'],
                                              win_shift=config['win_shift'],
                                              nDFT=config['nDFT'])

                save_path = os.path.join(config['save_testing_results_dir'],
                                         'test',
                                         str(config['test_noise'][noise_idx]),
                                         str(snr[snr_idx]))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                comp_save_path = os.path.join(save_path,
                                              os.path.basename(testing_file))
                output_data.save_wav_file(comp_save_path, rec_wav, config['sampling_rate'])

                save_path = os.path.join(config['save_testing_results_dir'],
                                         'mix',
                                         str(config['test_noise'][noise_idx]),
                                         str(snr[snr_idx]))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                comp_save_path = os.path.join(save_path,
                                              os.path.basename(testing_file))
                output_data.save_wav_file(comp_save_path, noisy_wav, config['sampling_rate'])

                save_path = os.path.join(config['save_testing_results_dir'],
                                         'clean',
                                         str(config['test_noise'][noise_idx]),
                                         str(snr[snr_idx]))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                comp_save_path = os.path.join(save_path,
                                              os.path.basename(testing_file))
                output_data.save_wav_file(comp_save_path, utter_wav, config['sampling_rate'])

                print("Testing file #", file_idx, os.path.basename(testing_file),
                      "SNR :", format(snr[snr_idx], '5.1f'), " "*10,
                      "noise:", format(config['test_noise'][noise_idx], '10.10s'), " "*10,
                      "mse:", format(test_mse, '.5f'))

                overall_testing_mse[snr_idx][noise_idx] = test_mse / (len(testing_file_list))

    np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

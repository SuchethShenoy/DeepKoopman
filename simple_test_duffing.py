import copy
import numpy as np
import tensorflow as tf
import simple_networkarch
import simple_training

def main():

    params = {}
    params['data_name'] = 'Duffing_uniformRandom'
    params['len_time'] = 800
    n = 2   # number of states
    k = 25  # number of lifted states
    num_initial_conditions = 700   # per training file
    # params['delta_t'] = 0.02        # check if it affects model
    params['folder_name'] = 'exp/exp_duffing_uniformRandom_2'

    params['num_shifts'] = 399  #params['len_time'] - 1
    params['num_shifts_middle'] = params['len_time'] - 1
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
    params['recon_lam'] = .001
    params['L1_lam'] = 0.0
    params['auto_first'] = 1

    params['num_passes_per_file'] = 15 * 6 * 50
    params['num_steps_per_batch'] = 2
    params['learning_rate'] = 10 ** (-3)

    params['max_time'] = 1 * 60 * 60  # 1 hours
    params['min_5min'] = .25
    params['min_20min'] = .02
    params['min_40min'] = .002
    params['min_1hr'] = .0002
    params['min_2hr'] = .00002
    params['min_3hr'] = .000004
    params['min_4hr'] = .0000005
    params['min_halfway'] = 1

    params['data_train_len'] = 5
    params['batch_size'] = int(2 ** 8)
    steps_to_see_all = num_examples / params['batch_size']
    print(f"steps to see all: {steps_to_see_all}")
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    params['L2_lam'] = 10 ** -14
    params['Linf_lam'] = 10 ** -8

    params['encoder_widths'] = [n, 128, 256, k]
    params['encoder_act_type'] = 'sigmoid'
    params['decoder_widths'] = [k, 256, 128, n]
    params['decoder_act_type'] = 'sigmoid'
    params['use_bias'] = False
    params['k_widths'] = [k, 64, 128, 64, k]

    simple_training.main_exp(copy.deepcopy(params))

if __name__ == '__main__':
    main()

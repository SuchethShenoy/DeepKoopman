import copy
import numpy as np
import tensorflow as tf
import simple_networkarch
import simple_training

def individual_test():
    # inp = np.array([[1, 1]], dtype='float64')
    encoder_widths = [2, 10, 5]
    x, weights, biases = simple_networkarch.encoder(encoder_widths, num_shifts_max=10)
    print(x, weights, biases)
    num_encoder_weights = len(weights)
    # print(simple_networkarch.encoder_apply_one_shift(inp, weights, biases, act_type='relu', num_encoder_weights=num_encoder_weights))


    g_list = simple_networkarch.encoder_apply(x, weights, biases, act_type='relu', shifts_middle=[1, 2],
                            num_encoder_weights=num_encoder_weights)
    print(g_list)

    k_block_widhts = [5, 8, 8, 5]
    weights = simple_networkarch.k_block(k_block_widhts)
    print(weights)

    y = np.array([[1,1,1,1,1]], dtype='float64')
    y_next = simple_networkarch.k_block_apply(g_list, weights, 5)
    print(y_next)

    decoder_widths = [5, 10, 2]
    weights, biases = simple_networkarch.decoder(decoder_widths)
    print(x, weights, biases)
    num_decoder_weights = len(weights)

    outputs = simple_networkarch.decoder_apply(y_next, weights, biases, act_type='relu', num_decoder_weights=num_decoder_weights)
    print(outputs)

def model_test():
    params = {
        'encoder_widths' : [2, 10, 5],
        'max_shifts_to_stack' : 10,
        'encoder_act_type' : 'sigmoid',
        'shifts_middle' : [1,2,3],
        'decoder_widths' : [5, 10, 2],
        'decoder_act_type' : 'linear',
        'k_widths' : [5, 8, 8, 5],
        'shifts' : [1,2,3,4]
    }

    x, y, g_list, weights, biases = simple_networkarch.create_koopman_net(params)

    print(f"x: {x}")
    print(f"y: {y}")
    print(f"g_list: {g_list}")
    print(f"weights: {weights}")
    print(f"biases: {biases}")

def main():

    params = {}
    params['data_name'] = 'Pendulum'
    params['len_time'] = 51
    n = 2   # number of states
    k = 12  # number of elevated states
    num_initial_conditions = 5000   # per training file
    params['delta_t'] = 0.02        # check if it affects model
    params['folder_name'] = 'exp/exp_pendulum'

    params['num_shifts'] = 30
    params['num_shifts_middle'] = params['len_time'] - 1
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
    params['recon_lam'] = .001
    params['L1_lam'] = 0.0
    params['auto_first'] = 1

    params['num_passes_per_file'] = 15 * 6 * 50
    params['num_steps_per_batch'] = 2
    params['learning_rate'] = 10 ** (-3)

    params['max_time'] = 0.5 * 60 * 60  # Changed 6 to 0.5 hour
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
    params['decoder_widths'] = [k, 256, 128, n]
    params['k_widths'] = [k, 64, 128, 64, k]

    simple_training.main_exp(copy.deepcopy(params))

if __name__ == '__main__':
    main()

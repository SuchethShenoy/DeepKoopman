import numpy as np
import tensorflow as tf
import simple_networkarch

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
    encoder_widths = [2, 10, 5]
    max_shifts_to_stack = 10
    encoder_act_type = 'sigmoid'
    shifts_middle = [1,2,3]
    decoder_widths = [5, 10, 2]
    decoder_act_type = 'linear'
    k_widths = [5, 8, 8, 5]
    shifts = [1,2,3,4]

    x, y, g_list, weights, biases = simple_networkarch.create_koopman_net(encoder_widths, max_shifts_to_stack, encoder_act_type, shifts_middle,
                                                                          decoder_widths, decoder_act_type, k_widths, shifts)

    print(f"x: {x}")
    print(f"y: {y}")
    print(f"g_list: {g_list}")
    print(f"weights: {weights}")
    print(f"biases: {biases}")

if __name__ == '__main__':
    model_test()

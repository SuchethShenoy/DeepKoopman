import numpy as np
import tensorflow as tf
import simple_networkarch

# inp = np.array([[1, 1]], dtype='float64')
encoder_widths = [2, 10, 5]
x, weights, biases = simple_networkarch.encoder(encoder_widths, num_shifts_max=10)
num_encoder_weights = len(weights)
# print(simple_networkarch.encoder_apply_one_shift(inp, weights, biases, act_type='relu', num_encoder_weights=num_encoder_weights))


g_list = simple_networkarch.encoder_apply(x, weights, biases, act_type='relu', shifts_middle=[1, 2],
                        num_encoder_weights=num_encoder_weights)

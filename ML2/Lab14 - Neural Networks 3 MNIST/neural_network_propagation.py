import numpy as np
import neural_network_setup
import pickle

def back_propagation_multilayer_MNIST(actual_table, num_epochs, num_weights, activation_function, activation_function_prime, test_network, cur_network, cur_network_accuracies, cur_network_errorsTotal, cur_network_errorsTotal_sum, cur_w_h1i_vals, cur_w_h1i_gradient_vals, cur_lambda_vals):                    #network = [ layer1, layer2, ... ]; layer1 = [ weight_matrix, bias_scalar ]              #actual_table = [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    epoch_count, cur_lamb = 0, cur_lambda_vals[-1]
    while(epoch_count < num_epochs):
        print(f"EPOCH {epoch_count}")
        cur_network_accuracies.append(test_network(cur_network, actual_table, activation_function))
        print(f"Network_Acc: {cur_network_accuracies[-1]}")
        epoch_error_totals = []
        for tup in actual_table:
            x, expected_output = tup
            #Forward Propagate
            a_vec, dot_vec, dot_vecs_ls, a_vecs_ls = neural_network_setup.forward_propagate(x, cur_network, activation_function)                            #a_vec = final layer's output perceptron, dot_vec = final layer's dot
            epoch_error_totals.append(neural_network_setup.calc_error(np.array([a_vec]), np.array([expected_output]), 10))
            cur_network_errorsTotal.append(epoch_error_totals[-1])
            #Backward Propagate - Calculate Gradient Descent Values
            delL_ls = [activation_function_prime(dot_vec)*(expected_output-a_vec)]                                              #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in range(len(cur_network)-2, -1, -1):
                delL_vector = activation_function_prime(dot_vecs_ls[l+1]) * ((cur_network[l+1][0]).T @ delL_ls[0])
                delL_ls = [delL_vector] + delL_ls
            #Keeping track of w1, w1_gradient, and lambda
            for i in range(5): cur_w_h1i_vals[i].append(cur_network[0][0][0][i])                                    #[ [1st iteration wh1_1, 2nd iteration wh1_1, ...], [1st iteration wh1_2, 2nd iteration wh1_2, ...], ... ]; list of size(# of instances)
            for i in range(5): cur_w_h1i_gradient_vals[i].append((delL_ls[0] @ (a_vecs_ls[0]).T)[0][i])             #[ [1st iteration dError/wh1_1, 2nd iteration dError/wh1_1, ...], [1st iteration dError/wh1_2, 2nd iteration dError/wh1_2, ...], ... ]; list of size(# of instances)
            cur_lambda_vals.append(cur_lamb)
            '''print(f"W1: {cur_w1_vals[-1]}")
            print(f"W1_Gradient: {cur_w1_gradient_vals[-1]}")
            print(f"Lambda: {cur_lambda_vals[-1]}")'''
            #Backward Propagate - Update Values
            for l in range(len(cur_network)):
                layer = cur_network[l]
                layer[1] = layer[1] + np.array([[cur_lamb]]) * delL_ls[l]                               #update bias
                layer[0] = layer[0] + np.array([[cur_lamb]]) * (delL_ls[l] @ (a_vecs_ls[l]).T)              #update weight
        cur_network_errorsTotal_sum.append(sum(epoch_error_totals))
        print(f"Epoch_MSE_Sum: {cur_network_errorsTotal_sum[-1]}")
        epoch_count += 1
        cur_lamb *= 0.99
        pickle.dump([cur_network, cur_network_accuracies, cur_network_errorsTotal, cur_network_errorsTotal_sum, cur_w_h1i_vals, cur_w_h1i_gradient_vals, cur_lambda_vals], open('network2.pkl', 'wb'))
        print("\n---------------------------\n")
        input()
    #return network, network_accuracies, network_errorsTotal, w1_vals, w1_gradient_vals, lambda_vals
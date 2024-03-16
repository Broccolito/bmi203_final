from nn import nn
from nn import preprocess

import numpy as np

def test_single_forward():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=492357816, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                          leniency = 1, progress = 1)
    W_curr = np.array([[0.7, 0.3], [0.982634, 0.2]])
    b_curr = np.array([[0.24], [0.2623]])
    A_prev = np.array([[0.245, 0.621]])
    activation = 'relu'
    A_curr, Z_curr = NN._single_forward(W_curr, b_curr, A_prev, activation)

    print(A_curr)
    print(Z_curr)
    # assert np.allclose(A_curr, [[0.5978, 0.62724533]]), "A_curr doesn't match expected result"
    # assert np.allclose(Z_curr, [[0.5978, 0.62724533]]), "Z_curr doesn't match expected result"

def test_forward():
    nn_arch_example = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    NN = nn.NeuralNetwork(nn_arch_example, 
                          lr=0.01, seed=492357816, batch_size=1, epochs=1, loss_function='mean_squared_error',
                          leniency = 1, progress = 1)
    X = np.array([[0.34321, 0.532]])
    W1 = np.array([[0.4325, 0.3695], [0.6, 0.12]])
    b1 = np.array([[0.21435], [0.34351]])
    W2 = np.array([[0.3532, 0.87235]])
    b2 = np.array([[0.31453]])
    NN._param_dict = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    output, cache = NN.forward(X)
    expected_output = np.array([[0.74021534]])
    print(output)
    # assert np.allclose(output, expected_output), "Output doesn't match expected result"
    
def test_single_backprop():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}],
                           lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                           leniency = 1, progress = 1)
    W_curr = np.array([[0.2141, 0.2343], [0.214, 0.51]])
    b_curr = np.array([[0.124], [0.325]])
    Z_curr = np.array([[0.532, 0.351]])
    A_prev = np.array([[0.87124, 0.214]])
    dA_curr = np.array([[1, 1]])
    activation_curr = 'relu'
    dA_prev, dW_curr, db_curr = NN._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
    # print(dA_prev, dW_curr, db_curr)
    print(dW_curr[0][0])
    # assert np.allclose(dW_curr[0][0], 0.87124), "dW_curr doesn't match expected result"
    # assert np.allclose(db_curr, [[1, 1]]), "db_curr doesn't match expected result"
    # assert np.allclose(dA_prev, [[0.4281, 0.7443]]), "dA_prev doesn't match expected result"

def test_predict():
    nn_arch_example = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    NN = nn.NeuralNetwork(nn_arch_example, 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                          leniency = 1, progress = 1)
    X = np.array([[0.5, 0.6]])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([[0.1], [0.2]])
    W2 = np.array([[0.5, 0.6]])
    b2 = np.array([[0.3]])
    NN._param_dict = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    pred = NN.predict(X)
    bi_pred = (pred > 0.5).astype(int)
    expected_output = np.array([[1]])
    print(pred)
    # assert np.allclose(bi_pred, expected_output), "Output doesn't match expected result"

def test_binary_cross_entropy():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy', 
                          leniency = 1, progress = 1)
    y_hat = np.array([[0.5], [0.92]])
    y = np.array([[1], [0]])
    loss = NN._binary_cross_entropy(y, y_hat)
    expected_loss = 1.6094379124341007
    print(loss)
    # assert np.allclose(loss, expected_loss), "Loss doesn't match expected result"

def test_binary_cross_entropy_backprop():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy', 
                          leniency = 1, progress = 1)
    y_hat = np.array([[0.8], [0.4]])
    y = np.array([[1], [0]])
    dA = NN._binary_cross_entropy_backprop(y, y_hat)
    expected_dA = [[-0.625],
                   [0.8333333]]
    print(dA)
    # assert np.allclose(dA, expected_dA), "dA doesn't match expected result"
    
def test_mean_squared_error():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error',
                            leniency = 1, progress = 1)
    y_hat = np.array([[0.6], [0.4]])
    y = np.array([[1], [0]])
    loss = NN._mean_squared_error(y, y_hat)
    expected_loss = 0.16000000000000003
    print(loss)
    # assert np.allclose(loss, expected_loss), "Loss doesn't match expected result"

def test_mean_squared_error_backprop():
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error',
                            leniency = 1, progress = 1)
    y_hat = np.array([[0.6], [0.4]])
    y = np.array([[1], [0]])
    dA = NN._mean_squared_error_backprop(y, y_hat)
    expected_dA = [[-0.8],
                   [ 0.8]]
    print(dA)
    # assert np.allclose(dA, expected_dA), "dA doesn't match expected result"
    
def test_sample_seqs():
    # Example sequences and labels (True for positive class, False for negative)
    seqs = ['ATGC', 'ATGG', 'TTAA', 'CCGG']  # Example sequences
    labels = [True, False, False, False]  # Corresponding labels

    # Call your sampling function
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    # Check that the number of True and False labels is balanced
    num_true = sum(sampled_labels)
    num_false = len(sampled_labels) - num_true

    # assert num_true == num_false, "The classes are not balanced"
    # assert len(sampled_seqs) == len(sampled_labels), "The number of sequences and labels does not match"
    # assert all(seq in seqs for seq in sampled_seqs), "Sampled sequences are not from the original set"

def test_one_hot_encode_seqs():
    # Example sequences
    seqs = ['ATGC', 'ATGG']

    # Expected one-hot encoding
    # A: [1, 0, 0, 0], T: [0, 1, 0, 0], C: [0, 0, 1, 0], G: [0, 0, 0, 1]
    expected_encoding = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # ATGC
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]   # ATGG
    ])

    # Call your one-hot encoding function
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)
    print(encoded_seqs)
    # Check that the encoding matches the expected result
    # assert np.array_equal(encoded_seqs, expected_encoding), "One-hot encoding does not match expected result"


# test_single_forward()
# test_forward()
test_single_backprop()
# test_predict()
# test_binary_cross_entropy()
# test_binary_cross_entropy_backprop()
# test_mean_squared_error()
# test_mean_squared_error_backprop()
# test_sample_seqs()
# test_one_hot_encode_seqs()
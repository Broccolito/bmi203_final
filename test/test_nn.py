# Import required modules and functions
from nn import nn
from nn import preprocess
import numpy as np

# Test function for a single forward pass in the neural network
def test_single_forward():
    # Initialize a neural network with a specific architecture and settings
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=492357816, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                          leniency=1, progress=1)
    # Set up test weights, biases, and input for the forward pass
    W_curr = np.array([[0.7, 0.3], [0.982634, 0.2]])
    b_curr = np.array([[0.24], [0.2623]])
    A_prev = np.array([[0.245, 0.621]])
    activation = 'relu'
    # Perform a single forward pass and capture the activation and pre-activation values
    A_curr, Z_curr = NN._single_forward(W_curr, b_curr, A_prev, activation)

    # Print the results and check if they match expected outcomes
    print(A_curr)
    print(Z_curr)
    assert np.allclose(A_curr, [[0.5978, 0.62724533]]), "A_curr doesn't match expected result"
    assert np.allclose(Z_curr, [[0.5978, 0.62724533]]), "Z_curr doesn't match expected result"

# Test function for a complete forward pass through the neural network
def test_forward():
    # Define the neural network architecture
    nn_arch_example = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    # Initialize the neural network with the specified architecture and settings
    NN = nn.NeuralNetwork(nn_arch_example, 
                          lr=0.01, seed=492357816, batch_size=1, epochs=1, loss_function='mean_squared_error',
                          leniency=1, progress=1)
    # Set up the input data and test parameters (weights and biases)
    X = np.array([[0.34321, 0.532]])
    W1 = np.array([[0.4325, 0.3695], [0.6, 0.12]])
    b1 = np.array([[0.21435], [0.34351]])
    W2 = np.array([[0.3532, 0.87235]])
    b2 = np.array([[0.31453]])
    # Manually set the parameters in the neural network
    NN._param_dict = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    # Perform a forward pass with the input data
    output, cache = NN.forward(X)
    # Define the expected output for comparison
    expected_output = np.array([[0.74021534]])
    # Print the actual output and verify it matches the expected output
    print(output)
    assert np.allclose(output, expected_output), "Output doesn't match expected result"

# Test function for a single backward propagation step
def test_single_backprop():
    # Initialize a neural network with a specific architecture and settings for testing backpropagation
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}],
                           lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                           leniency=1, progress=1)
    # Define weights, biases, activations, and gradients for the backpropagation test
    W_curr = np.array([[0.2141, 0.2343], [0.214, 0.51]])
    b_curr = np.array([[0.124], [0.325]])
    Z_curr = np.array([[0.532, 0.351]])
    A_prev = np.array([[0.87124, 0.214]])
    dA_curr = np.array([[1, 1]])
    activation_curr = 'relu'
    # Perform a single backpropagation step
    dA_prev, dW_curr, db_curr = NN._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

    # Print the gradients for inspection and verify against expected values
    print(dA_prev, dW_curr, db_curr)
    assert np.allclose(dW_curr[0][0], 0.87124), "dW_curr doesn't match expected result"
    assert np.allclose(db_curr, [[1, 1]]), "db_curr doesn't match expected result"
    assert np.allclose(dA_prev, [[0.4281, 0.7443]]), "dA_prev doesn't match expected result"

# Test function for predicting output with the neural network
def test_predict():
    # Define the neural network architecture for prediction
    nn_arch_example = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    # Initialize the neural network with the specified architecture and settings
    NN = nn.NeuralNetwork(nn_arch_example, 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error', 
                          leniency=1, progress=1)
    # Set up the input data and neural network parameters for prediction
    X = np.array([[0.5, 0.6]])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([[0.1], [0.2]])
    W2 = np.array([[0.5, 0.6]])
    b2 = np.array([[0.3]])
    # Manually set the parameters in the neural network
    NN._param_dict = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    # Use the neural network to predict the output for the input data
    pred = NN.predict(X)
    # Convert the continuous prediction to binary prediction based on a threshold
    bi_pred = (pred > 0.5).astype(int)
    # Define the expected binary output for comparison
    expected_output = np.array([[1]])
    # Print the prediction and verify it matches the expected binary output
    print(pred)
    assert np.allclose(bi_pred, expected_output), "Output doesn't match expected result"

# Test function for calculating binary cross entropy loss
def test_binary_cross_entropy():
    # Initialize a neural network with specific settings for testing binary cross entropy loss calculation
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy', 
                          leniency=1, progress=1)
    # Define predicted and actual values for the loss calculation
    y_hat = np.array([[0.5], [0.92]])
    y = np.array([[1], [0]])
    # Calculate the binary cross entropy loss
    loss = NN._binary_cross_entropy(y, y_hat)
    # Define the expected loss value for comparison
    expected_loss = 1.6094379124341007
    # Print the calculated loss and verify it matches the expected loss
    print(loss)
    assert np.allclose(loss, expected_loss), "Loss doesn't match expected result"

# Test function for backpropagation with binary cross entropy loss
def test_binary_cross_entropy_backprop():
    # Initialize a neural network with specific settings for testing backpropagation with binary cross entropy loss
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy', 
                          leniency=1, progress=1)
    # Define predicted and actual values for calculating the gradient of the loss
    y_hat = np.array([[0.8], [0.4]])
    y = np.array([[1], [0]])
    # Calculate the gradient of the loss with respect to the activation
    dA = NN._binary_cross_entropy_backprop(y, y_hat)
    # Define the expected gradient values for comparison
    expected_dA = [[-0.625],
                   [0.8333333]]
    # Print the calculated gradients and verify they match the expected values
    print(dA)
    assert np.allclose(dA, expected_dA), "dA doesn't match expected result"

# Test function for calculating mean squared error loss
def test_mean_squared_error():
    # Initialize a neural network with specific settings for testing mean squared error loss calculation
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error',
                          leniency=1, progress=1)
    # Define predicted and actual values for the loss calculation
    y_hat = np.array([[0.6], [0.4]])
    y = np.array([[1], [0]])
    # Calculate the mean squared error loss
    loss = NN._mean_squared_error(y, y_hat)
    # Define the expected loss value for comparison
    expected_loss = 0.16000000000000003
    # Print the calculated loss and verify it matches the expected loss
    print(loss)
    assert np.allclose(loss, expected_loss), "Loss doesn't match expected result"

# Test function for backpropagation with mean squared error loss
def test_mean_squared_error_backprop():
    # Initialize a neural network with specific settings for testing backpropagation with mean squared error loss
    NN = nn.NeuralNetwork([{'input_dim': 2, 'output_dim': 32, 'activation': 'relu'}], 
                          lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_squared_error',
                          leniency=1, progress=1)
    # Define predicted and actual values for calculating the gradient of the loss
    y_hat = np.array([[0.6], [0.4]])
    y = np.array([[1], [0]])
    # Calculate the gradient of the loss with respect to the activation
    dA = NN._mean_squared_error_backprop(y, y_hat)
    # Define the expected gradient values for comparison
    expected_dA = [[-0.8],
                   [ 0.8]]
    # Print the calculated gradients and verify they match the expected values
    print(dA)
    assert np.allclose(dA, expected_dA), "dA doesn't match expected result"

# Test function for sampling sequences to balance classes
def test_sample_seqs():
    # Define example sequences and corresponding labels
    seqs = ['ATGC', 'ATGG', 'TTAA', 'CCGG']  # Example sequences
    labels = [True, False, False, False]  # Corresponding labels

    # Perform class balancing by sampling sequences
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    # Check the balance between classes and consistency of the sampling
    num_true = sum(sampled_labels)
    num_false = len(sampled_labels) - num_true
    assert num_true == num_false, "The classes are not balanced"
    assert len(sampled_seqs) == len(sampled_labels), "The number of sequences and labels does not match"
    assert all(seq in seqs for seq in sampled_seqs), "Sampled sequences are not from the original set"

# Test function for one-hot encoding of sequences
def test_one_hot_encode_seqs():
    # Define example sequences to be encoded
    seqs = ['ATGC', 'ATGG']

    # Define the expected one-hot encoding for the sequences
    # A: [1, 0, 0, 0], T: [0, 1, 0, 0], C: [0, 0, 1, 0], G: [0, 0, 0, 1]
    expected_encoding = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # ATGC
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]   # ATGG
    ])

    # Perform one-hot encoding on the example sequences
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)
    # Print the encoded sequences for inspection
    print(encoded_seqs)
    # Verify the one-hot encoding matches the expected result
    assert np.array_equal(encoded_seqs, expected_encoding), "One-hot encoding does not match expected result"

import random
from equation_tree import instantiate_constants
from equation_tree.sample import sample
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from equation_tree.defaults import DEFAULT_PRIOR
from equation_tree.prior import priors_from_space

prior = DEFAULT_PRIOR
#prior['functions'] = priors_from_space(['sin', 'cos', 'tan', 'exp', 'log'])
#prior['operators'] = priors_from_space(['+', '-', '*', '**', '/'])

# Dataset n-hot encoded equations elements
class EquationDatasetNew(Dataset):
    def __init__(self, equations, conditioning_values, char_to_idx, constants):
        self.equations = equations
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        self.conditioning_values = conditioning_values
        self.constants = constants

    def __len__(self):
        return len(self.equations)

    def encode_equation(self, equation):
        encoded = []
        for element in equation:
            if element in self.char_to_idx:
                encoded.append(self.char_to_idx[element])
            else:
                # Handle unknown elements or special cases
                encoded.append(self.char_to_idx["<PAD>"])
        return encoded

    def decode_equation(self, encoded):
        encoded_flatten = encoded.flatten()
        equation = []
        for element in encoded_flatten:
            equation.append(self.idx_to_char[int(element)])
        return equation

    def __getitem__(self, idx):
        equation = self.equations[idx]
        equation_encoded = self.encode_equation(equation)  # [idx]
        conditioning_values = self.conditioning_values[idx]
        constants = self.constants[idx]
        # print(equation_encoded)
        return (
            torch.tensor(equation_encoded, dtype=torch.long),
            torch.tensor(constants, dtype=torch.float),
            torch.tensor(conditioning_values, dtype=torch.float),
        )

# Dataset one-hot encoded function terms
class EquationDatasetClassify(Dataset):
    def __init__(self, equations, conditioning_values, char_to_idx, num_classes, constants):
        self.equations = equations
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        self.conditioning_values = conditioning_values
        self.constants = constants
        self.num_classes = num_classes

    def __len__(self):
        return len(self.equations)

    def encode_equation(self, equation):
        encoded = []
        for element in equation:
            if element in self.char_to_idx:
                num_encoding = self.char_to_idx[element]
                # one hot encoding
                encoded.append(num_encoding)
            else:
                # Handle unknown elements or special cases
                encoded.append(self.char_to_idx["<PAD>"])
        one_hot_equation = []
        for x in range(len(encoded)):
            one_hot_char = [0 for _ in range(self.num_classes)]
            one_hot_char[x] = 1
            one_hot_equation.append(one_hot_char)
        return encoded

    def decode_equation(self, encoded):
        # argmax to get the index of the one hot encoding
        #encoded = torch.argmax(encoded, dim=1)
        #encoded = np.argmax(encoded, axis=1)
        #encoded_flatten = encoded.flatten()
        equation = []
        for element in encoded:
            equation.append(self.idx_to_char[int(element)])
        return equation

    def __getitem__(self, idx):
        equation = self.equations[idx]
        equation_encoded = self.encode_equation(equation)  # [idx]
        conditioning_values = self.conditioning_values[idx]
        constants = self.constants[idx]
        # print(equation_encoded)
        return (
            torch.tensor(equation_encoded, dtype=torch.long),
            torch.tensor(constants, dtype=torch.float),
            torch.tensor(conditioning_values, dtype=torch.float),
        )


def generate_dataset(num_equation_samples):
    """
    Generate a dataset n-hot encoded equations elements.

    Parameters:
    - num_equation_samples (int): Number of equations to sample.

    Returns:
    - dataset (EquationDatasetNew): Dataset containing n-hot encoded equations elements.
    - max_len (int): Maximum length of the equations.
    - unique_symbols (list): List of unique symbols in the dataset.
    """
    equations = []
    equations_final = []
    values = []
    constants = []
    max_len = 0

    i = 0
    while i < num_equation_samples - 2:
        # sample equation
        equation = sample(max_num_variables=1)
        # only add equation if it has maximum one constant
        if equation[0].n_constants == 1:
            # add equation 3 times (with different constants later)
            try:
                for _ in range(3):
                    # calculate maximum length of equation for padding later
                    if len(equation[0].prefix) > max_len:
                        max_len = len(equation[0].prefix)
                    # instantiate constant randomly
                    instantiated_equation = instantiate_constants(
                        equation[0], lambda: random.random()
                    )
                    # evaluate the equation at 50 equally spaced points between -1 and 1
                    x_1 = np.linspace(-1, 1, 50)
                    input_df = pd.DataFrame({"x_1": x_1.tolist()})

                    # get f(x) values
                    y = instantiated_equation.evaluate(input_df)
                    # check if there are nan values in y
                    if all(val == val for val in y) and max(y) < 1000:
                        const = instantiated_equation.constants
                        constants.append([float(const[0])])
                        values.append((input_df["x_1"].values, y))
                        equations.append(equation)
                        i += 1
            except Exception as e:
                # catch exception in case instantiate_constants() throws 'ComplexInfinity' Exception
                print(e)
                print(equation)

    for equation in equations:
        # try block due to complex infinity exception
        eq_prefix = equation[0].prefix
        # add padding so that all equations have the same shape
        if len(eq_prefix) < max_len:
            eq_prefix = eq_prefix + ["<PAD>"] * (max_len - len(eq_prefix))
        # add equations, constants and values to their list
        equations_final.append(eq_prefix)

    all_symbols = [item for sublist in equations_final for item in sublist]
    unique_symbols = sorted(list(set(all_symbols)))

    # obtain mapping from symbols to indices and vice versa
    symb_to_idx = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    idx_to_symb = {idx: symb for symb, idx in symb_to_idx.items()}

    dataset = EquationDatasetNew(equations_final, values, symb_to_idx, constants)
    # pd.DataFrame(values, columns).to_csv('values_data.csv')
    return dataset, max_len, unique_symbols


def preprocessing(dataset, batch_size, training_set_proportion):
    """
    Preprocess the dataset and create data loaders.

    Parameters: 
    - dataset (EquationDatasetNew): Dataset to preprocess.
    - batch_size (int): Number of samples per batch.
    - training_set_proportion (float): Proportion of the dataset to include in the training set.

    Returns:
    - train_dataloader (DataLoader): DataLoader for the training set.
    - test_dataloader (DataLoader): DataLoader for the test set.
    - len(test_dataset) (int): Number of samples in the test set.
    """
    # split data into training and test sets
    train_size = int(training_set_proportion * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    return train_dataloader, test_dataloader, len(test_dataset)

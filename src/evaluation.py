import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
from equation_tree.util.conversions import prefix_to_infix
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from equation_tree.util.conversions import prefix_to_infix
import torch
import pandas as pd
from plotly.offline import plot
from colour import Color
from src.loss import correlation_coeff
from src.utils import try_infix
#from src.utils import is_function, is_operator, is_variable
from equation_tree import instantiate_constants
from equation_tree.util.conversions import infix_to_prefix
from equation_tree.tree import node_from_prefix
from equation_tree.tree import EquationTree
from src.preprocessing import preprocessing
from src.models import vae_ecv_loss, ae_ec_loss
from IPython.utils import io
from src.training import training_VAE, training_AE, training_AE_C, training_VAE_C
from sympy import *


def evaluation_ec(x_batches, x_hat_batches, equation_tree_dataset, max_len, kind):
    """
    Evaluate the performance of the model

    Parameters:
        x_batches: oroginal equations
        x_hat_batches: reconstructed equations
        equation_tree_dataset: dataset
        max_len: maximum length of the equations
        kind: kind of model (AE, VAE, AE_C, VAE_C)
    
    Returns:
        results, x_batches_p, x_hat_batches_p, x_constants_p, x_hat_constants_p
    """
    x_batches_p = [x.detach().numpy() for sublist in x_batches for x in sublist[0]]
    if kind != 'AE_C' and kind != 'VAE_C':
        x_hat_batches_p = [
            x.detach().numpy() for sublist in x_hat_batches for x in sublist[0]
        ]
        x_hat_constants_p = [
            x.detach().numpy() for sublist in x_hat_batches for x in sublist[1]
        ]
    else:
        x_hat_batches_p = [
            x for sublist in x_hat_batches[0] for x in sublist
        ]
        x_hat_constants_p = [
            x for sublist in x_hat_batches[1] for x in sublist
        ]
    x_constants_p = [x[0].detach().numpy() for sublist in x_batches for x in sublist[1]]
    

    rand_idx1 = np.random.randint(0, len(x_batches_p))
    rand_idx2 = np.random.randint(0, len(x_batches_p))
    while rand_idx1 == rand_idx2:
        rand_idx2 = np.random.randint(0, len(x_batches_p))

    # reduce dimensionality of x hat by taking the mean
    # get a prediction by selecting the index with highest probability
    if kind != 'AE_C' and kind != 'VAE_C':
        x_hat_batches_p = np.argmax(x_hat_batches_p, axis=2)

    # How do you get the prediction for the constants?
    #x_hat_constants_p = np.mean(x_hat_constants_p, axis=1)
    x_decoded1 = equation_tree_dataset.decode_equation(x_batches_p[rand_idx1])
    x_decoded2 = equation_tree_dataset.decode_equation(x_batches_p[rand_idx2])
    x_hat_decoded = equation_tree_dataset.decode_equation(x_hat_batches_p[rand_idx1])

    # Checks how many elements of the original equations were correctly recovered
    correctly_recovered = 0
    for x, x_hat in zip(x_batches_p, x_hat_batches_p):
        for e, e_hat in zip(x, x_hat):
            if e == e_hat:
                correctly_recovered += 1
            else:
                pass
                # print(x)
                # print(x_hat)

    full_recovered = 0

    if kind == 'AE_C' or kind == 'VAE_C':
        for x, x_hat in zip(x_batches_p, x_hat_batches_p):
            x = x.tolist()
            if x == x_hat:
                full_recovered += 1
        c_abs = [np.abs(c - c_hat.detach().numpy()) for c, c_hat in zip(x_constants_p, x_hat_constants_p)]
        mse_constants = [
            np.square(c - c_hat.detach().numpy()) for c, c_hat in zip(x_constants_p, x_hat_constants_p)
        ]
    else:
        for x, x_hat in zip(x_batches_p, x_hat_batches_p):
            x = x.tolist()
            if all(x == x_hat):
                full_recovered += 1
        c_abs = [np.abs(c - c_hat) for c, c_hat in zip(x_constants_p, x_hat_constants_p)]
        mse_constants = [
            np.square(c - c_hat) for c, c_hat in zip(x_constants_p, x_hat_constants_p)
        ]
    c_distance = np.sum(c_abs) / len(x_constants_p)

    # print mse constants
    
    mse_constants = np.sum(mse_constants) / len(x_constants_p)
    results = {
        "rand_idx1": rand_idx1,
        "rand_idx2": rand_idx2,
        "original rand_idx1": x_batches_p[rand_idx1],
        # "original rand_idx2": x_batches_p[rand_idx2],
        "reconstructed rand_idx1": str(x_hat_batches_p[rand_idx1]),
        # "reconstructed rand_idx2": x_hat_batches_p[rand_idx2],
        "x_decoded1": x_decoded1,
        "x_decoded2": x_decoded2,
        "x_hat_decoded": x_hat_decoded,
        "original constants rand_idx1": x_constants_p[rand_idx1],
        "original constants rand_idx2": x_constants_p[rand_idx2],
        "reconstructed constants rand_idx1": x_hat_constants_p[rand_idx1],
        # "reconstructed constants rand_idx2": x_hat_constants_p[rand_idx2],
        "correctly recovered individual signs": f"{correctly_recovered} out of {len(x_batches_p) * max_len}",
        "accuracy (individual)": correctly_recovered / (len(x_batches_p) * max_len),
        "correctly recovered equations": f"{full_recovered} out of {len(x_batches_p)}",
        "accuracy (equations)": full_recovered / len(x_batches_p),
        "average distance constants": c_distance,
        "average mse constants": mse_constants,
    }
    # df = pd.DataFrame(results)
    return results, x_batches_p, x_hat_batches_p, x_constants_p, x_hat_constants_p


def plot_functions(equations, values, constants, is_function, is_operator):
    """
    Plot the function graphs

    Parameters:
        equations: equations to plot
        values: values of the function graphs
        constants: constants
        is_function: function that checks if a symbol is a function
        is_operator: function that checks if a symbol is an operator 
    """
    if len(equations) == 0:
        return None
    colors = list(Color("violet").range_to(Color("green"), len(equations)))
    try:
        fig = go.Figure(
                data=go.Scatter(
                    x=values[0][0],
                    y=values[0][1],
                    mode="lines",
                    line = dict(color = colors[0].hex,
                                            width = 4), 
                    name=try_infix(equations[0], constants[0]),
                )
                
        )
        '''
        fig.add_trace(
            go.Bar(
                x=values[0][0],
                y=values[0][1],
                #mode="lines",
                name=try_infix(equations[0], constants[0]),
            )
        )
        '''
        for i in range(1, len(equations)):
            try: 
                fig.add_trace(
                    go.Scatter(
                        x=values[i][0],
                        y=values[i][1],
                        mode="lines",
                        line = dict(color = colors[i].hex,
                                            width = 4), 
                        name= try_infix(equations[i], constants[i]),
                    )
                )
            except Exception as e:
                print(f"failed to plot {equations[i]}")
                
        #plot(fig,filename="functions.html",auto_open=False,image='png')
        
    except IndexError:
        print("no functions to plot")
        fig = None
    return fig


def get_latent_representation(
    model,
    device,
    test_dataloader,
    x_batches_p,
    x_hat_batches_p,
    equation_tree_dataset,
):
    """
    Get the latent space representation of the model

    Parameters:
        model: model
        device: device
        test_dataloader: test dataloader
        x_batches_p: original equations
        x_hat_batches_p: reconstructed equations
        equation_tree_dataset: dataset

    Returns:
        latent_space_representation, x_batches_p, test_values
    """
    latent_space_representation = []
    # get random number between 0 and test_size
    test_values = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for (
            test_equations_batch,
            test_constant_batch,
            test_values_batch,
        ) in test_dataloader:
            test_equations_batch = test_equations_batch.to(device)  #
        
            out = model.encode(test_equations_batch, test_constant_batch)
            if len(out) == 2:
                mean, logvar = out
                z_syntax = model.reparameterize(mean, logvar)
            else:
                z_syntax = out
            # print(z_syntax)
            latent_space_representation.append(z_syntax.cpu().numpy())
            test_values += test_values_batch

    latent_space_representation = np.concatenate(latent_space_representation, axis=0)
    # take only the first two dimensions of the latent space
    lat = latent_space_representation  # [:, :2]

    return latent_space_representation, x_batches_p, test_values,

def get_interpolated_df(
    kind,
    model,
    equation_tree_dataset,
    latent_space_representation,
    equation_1,
    equation_2,
    c_1,
    c_2,
    num_interpolations,
    assignment,
    classes = None,):
    """
    Get the dataframe with the interpolated equations of equation_1 and equation_2 

    Parameters:
        kind: kind of model (AE, VAE, AE_C, VAE_C)
        model: model
        equation_tree_dataset: dataset
        latent_space_representation: latent space representation
        equation_1: first equation
        equation_2: second equation
        c_1: constant 1
        c_2: constant 2
        num_interpolations: number of interpolations
        assignment: assignment (is_function, is_operator, is_variable, is_constant)
        classes: classes
    
    Returns:
        df, z_list
    """
    is_function, is_operator, is_variable, is_constant = assignment
    # get 2 random vectors from the latent space
    if type(equation_1) == int:
        z1 = np.array(latent_space_representation[equation_1])
        z2 = np.array(latent_space_representation[equation_2])
    else: 
        equation_1 = infix_to_prefix(equation_1, is_function, is_operator)
        equation_2 = infix_to_prefix(equation_2, is_function, is_operator)
        
        if len(equation_1) < 6:
            equation_1 = equation_1 + ["<PAD>"] * (6 - len(equation_1))
        if len(equation_2) < 6:
            equation_2 = equation_2 + ["<PAD>"] * (6 - len(equation_2))
        print(equation_1)
        equation_1 = equation_tree_dataset.encode_equation(equation_1)
        equation_2 = equation_tree_dataset.encode_equation(equation_2)

        encoded_1 = model.encode(torch.tensor([equation_1]), torch.tensor([[c_1]]))
        encoded_2 = model.encode(torch.tensor([equation_2]), torch.tensor([[c_2]]))
        if len(encoded_1) == 2:
            mean1, logvar1 = encoded_1
            mean2, logvar2 = encoded_2
            z1 = model.reparameterize(mean1, logvar1).detach().numpy()
            z2 = model.reparameterize(mean2, logvar2).detach().numpy()
        else:
            z1 = encoded_1.detach().numpy()
            z2 = encoded_2.detach().numpy()
    # generate 3 vectors between z1 and z2
    coords = []
    for i in range(len(z1)):
        l = np.linspace(z1[i], z2[i], num_interpolations + 2, dtype=np.float32)[1:-1]
        coords.append(l)

    # create a list of 3 vectors
    z_list = []
    z_list.append(z1)
    for i in range(num_interpolations):
        z_list.append([coords[k][i] for k in range(len(z1))])
    z_list.append(z2)
    z_list = np.array(z_list)
    if len(z_list.shape) == 3:
        z_list = z_list.squeeze(1)
    if kind == 'regression':
        z_decoded_list, z_decoded_constants = decode_latent(model, equation_tree_dataset, z_list)
    else:
        z_decoded_list, z_decoded_constants = decode_latent_classify(model, equation_tree_dataset, z_list, classes)
    print(f"reconstructed euqation 1: {z_decoded_list[0]}, reconstructed equation 2: {z_decoded_list[-1]}")

    # should I add correct or decoded equations?
    #z_decoded_list[0] = x_decoded_1
    #z_decoded_list[-1] = x_decoded_2

    df = pd.DataFrame(data=z_list[:, :3], columns=["x", "y", "z"])
    df["Category_prefix"] = [str(v) for v in z_decoded_list]
    df["constants"] = [v for v in z_decoded_constants]
    for i in range(len(z_decoded_list)):
        try:
            z_decoded_list[i] = prefix_to_infix(z_decoded_list[i], is_function, is_operator)
        except IndexError as e:
            print(e)
    df["Category"] = [str(v) for v in z_decoded_list]
    return df, z_list

def decode_latent(model, equation_tree_dataset, z_list):
    """
    Decode the latent space representation for the training_ecv approach

    Parameters:
        model: model
        equation_tree_dataset: dataset
        z_list: latent space representation
    
    Returns:
        z_decoded_equations, z_decoded_constants
    """

    z_decoded_equations = []
    z_decoded_constants = []
    for v in z_list:
        v = torch.tensor(2 * [v])
        v_decode, v_constants = model.decode(v)
        v_constants = v_constants.detach().numpy()
        v_decode = v_decode.detach().numpy()
        v_decode = np.argmax(v_decode, axis=2)
        v_decode = equation_tree_dataset.decode_equation(v_decode)
        z_decoded_equations.append(v_decode)
        z_decoded_constants.append(v_constants)
    
    return z_decoded_equations, z_decoded_constants

def decode_latent_classify(model, equation_tree_dataset, z_list, classes):
    """
    Decode the latent space representation for the training_ecv_classify approach

    Parameters:
        model: model
        equation_tree_dataset: dataset
        z_list: latent space representation
        classes: classes
    
    Returns:
        z_decoded_equations, z_decoded_constants
    """
    z_decoded_equations = []
    z_decoded_constants = []
    for v in z_list:
        v = torch.tensor(2 * [v])
        v_decode, v_constants = model.decode(v)
        v_constants = v_constants.detach().numpy()
        v_decode = v_decode.detach().numpy()
        v_decode = np.argmax(v_decode, axis=1)
        v_decode = [classes[i] for i in v_decode]
        v_decode = equation_tree_dataset.decode_equation(v_decode[0])
        z_decoded_equations.append(v_decode)
        z_decoded_constants.append(v_constants)
    
    return z_decoded_equations, z_decoded_constants


def get_correlation_coefficient(
    latent_space_representation,
    x_decoded,
    is_function,
    is_operator,
    x_constants_p,
    test_values,
    dataset,
):
    """
    Get the correlation coefficient between the latent space representation and the test values

    Parameters:
        latent_space_representation: latent space representation
        x_decoded: decoded equations
        is_function: function that checks if a symbol is a function
        is_operator: function that checks if a symbol is an operator
        x_constants_p: constants
        test_values: test values
        dataset: dataset
    
    Returns:
        correlation_cor: correlation between correlation coefficient, 
        correlation_dis: correlation between distance coefficient, 
        distance_matrix_lat: distance matrix between latent space representation, 
        distance_matrix_values: distance matrix between values of the function graphs, 
        df: dataframe for plotting, 
        test_values_det: test values, 
        dm_values: distance matrix between test values, 
        distance_df_values: distance dataframe between test values
    """
    # Calculate distance matrices

    df = pd.DataFrame(data=latent_space_representation[:, :3], columns=["x", "y", "z"])
    x_decoded = [dataset.decode_equation(eq) for eq in x_decoded]
    df["Category_prefix"] = [str(eq) for eq in x_decoded]
    df["Category"] = [prefix_to_infix(eq, is_function, is_operator) for eq in x_decoded]
    # replace c_1 with the actual constant
    df["Category"] = [
        eq.replace("c_1", str(round(float(c), 2)))
        for eq, c in zip(df["Category"], x_constants_p)
    ]

    correlation_cor, correlation_dis, _,_ = correlation_coeff(values=torch.stack(test_values)[:,1,:], z=torch.tensor(latent_space_representation))

    distance_matrix_lat = distance_matrix(
        latent_space_representation, latent_space_representation
    )
    distance_df_lat = pd.DataFrame(
        distance_matrix_lat, columns=df["Category"], index=df["Category"]
    )

    test_values_det = np.array([values.detach().numpy() for values in test_values])

    # create distance matrix where the values are the distance between each test values
    distance_matrix_values = np.zeros((len(test_values_det), len(test_values_det)))
    dm_values = np.zeros((len(test_values_det), len(test_values_det)))

    for i in range(len(test_values_det)):
        for j in range(len(test_values_det)):
            if i == j:
                distance_matrix_values[i][j] = 0.0
            else:
                distance_matrix_values[i][j] = np.nan_to_num(
                    np.mean(np.abs(test_values_det[i][1] - test_values_det[j][1]))
                )
                dm_values[i][j] = np.nan_to_num(
                    np.mean(test_values_det[i][1] - test_values_det[j][1])
                )
                # print(test_values_det[i][1])
    distance_matrix_values = np.nan_to_num(distance_matrix_values, posinf=1000, neginf=-1000)
    distance_matrix_lat = np.nan_to_num(distance_matrix_lat, posinf=1000, neginf=-1000)
    distance_df_values = pd.DataFrame(
        distance_matrix_values, columns=df["Category"], index=df["Category"]
    )

    #correlation_coefficient, p_value = pearsonr(
     #   squareform(distance_matrix_lat), squareform(distance_matrix_values)
    #)
    # print(f"Correlation coefficient: {correlation_coefficient}")
    return (
        correlation_cor,
        correlation_dis,
        distance_matrix_lat,
        distance_matrix_values,
        df,
        test_values_det,
        dm_values,
        distance_df_values,
    )


def plot_latent_vectors(z_list, interpolated_df, results, distance_matrix_lat, distance_matrix_values):
    """
    Plot the latent vectors

    Parameters:
        z_list: latent space representation
        interpolated_df: interpolated dataframe
        results: results dataframe
        distance_matrix_lat: distance matrix between latent space representation
        distance_matrix_values: distance matrix between values of the function graphs
    """
    colors = list(Color("green").range_to(Color("blue"), len(z_list)))
    vectors = []
    for i in range(len(z_list)):
        vector = go.Scatter3d( x = [0, z_list[i, 0]],
                            y = [0, z_list[i, 1]],
                            z = [0, z_list[i, 2]],
                            marker = dict( size = 0,             
                                            color = colors[i].hex,),
                            line = dict( color = colors[i].hex,
                                            width = 4), 
                            hoverinfo="x+y+z+text",
                            #legendgrouptitle=dict(text=str(interpolated_df['Category_prefix'][i])),
                            #legendgroup=str(interpolated_df['Category_prefix'][i]),
                            hovertext=str(interpolated_df['Category'][i]),
                            name=str(interpolated_df['Category'][i]),
                            )
        vectors.append(vector)

    arrows = []
    for i in range(len(z_list)):
        arrow = go.Cone( x = [z_list[i, 0]],
                        y = [z_list[i, 1]],
                        z = [z_list[i, 2]],
                        u = [z_list[i, 0]],
                        v = [z_list[i, 1]],
                        w = [z_list[i, 2]],
                        sizeref=0.1,
                        colorscale = [colors[i].hex,colors[i].hex],
                        showscale=False,
                        hoverinfo="none",
                        )
        arrows.append(arrow)
                        
    data = arrows + vectors

    fig = go.Figure(data=data, 
                    layout=go.Layout(title=f"Latent Space Representation <br><sup>Latent distance: {round(distance_matrix_lat[results['rand_idx1']][results['rand_idx2']],2)} Values distance: {round(distance_matrix_values[results['rand_idx1']][results['rand_idx2']],2)}</sup> ",
                        
                        ))
    fig.update_layout(legend = dict(font = dict(size = 10), #yanchor="top",
                                    orientation="h",
                                    yanchor="bottom",
                                    x=0
                                    ),
                    legend_title = dict(font = dict(size = 2)),

                    width=800, height=800)
    plot(fig,filename="vector.html",auto_open=False,image='png')
    fig.show()

def plot_interpolations(interpolated_df, assignment):
    """
    Plot the interpolations based on the interpolated dataframe 

    Parameters:
        interpolated_df: interpolated dataframe (get_interpolated_df())
        assignment: assignment (is_function, is_operator, is_variable, is_constant)
    """
    is_function= assignment[0]
    is_operator = assignment[1]
    is_var = assignment[2]
    is_con = assignment[3]
    colors = list(Color("green").range_to(Color("blue"), len(interpolated_df.values)))
    equations = []
    equations_infix = []
    constants = []
    values = []
    for equation in interpolated_df.values:
        value = generate_values(equation[5], equation[4][0][0], is_function, is_operator, is_var, is_con)
        equation_prefix = infix_to_prefix(equation[5], is_function, is_operator,)
        if len(value) == 2:
            equations.append(equation_prefix)
            equations_infix.append(equation[5])
            constants.append(equation[4][0][0])
            values.append(value)
    if len(equations) > 0:
        print(f"{len(equations)} out of {len(interpolated_df.values)} equations were successfully evaluated")
        #print(f"The two original functions ({equations_infix[0]} and ({equations_infix[-1]})) have a correlation of {np.corrcoef(values[0][1], values[-1][1])[0][1]}, a covariance of {np.cov(values[0][1], values[-1][1])[0][1]} and a distance of {np.linalg.norm(np.array(values[0][1]) - np.array(values[-1][1], ), ord=1)/25}")
        constants_f = [[float(c)] for c in constants]
        fig = plot_functions(equations=equations, constants=constants_f, values=values, is_function=is_function, is_operator=is_operator,)
        
    else:
        fig = None
    return fig

def generate_values(equation, constant, is_function, is_operator, is_variable, is_constant, infix=None):
    """
    Generate the values of the function graphs

    Parameters:
        equation: equation
        constant: constant
        is_function: function that checks if a symbol is a function
        is_operator: function that checks if a symbol is an operator
        is_variable: function that checks if a symbol is a variable
        is_constant: function that checks if a symbol is a constant
        infix: infix notation (in case it exists)
    
    Returns:
        x values, y values
    """
    #is_constant = lambda x: x in ["c_1"]
    constant = float(constant)
    if type(equation) == str:
        equation_prefix = infix_to_prefix(equation, is_function, is_operator,)
        
    else:
        equation_prefix = equation
        if infix == None:
            try:
                infix = prefix_to_infix(equation_prefix, is_function, is_operator)
            except:
                print(f"Failed to convert {equation_prefix} to infix")
                return (None,)
    try: 
        equation_node = node_from_prefix(equation_prefix, is_function, is_operator, is_variable, is_constant)
    except:
        print(f"Failed to create tree: {equation_prefix}")
        return (None,)
    equation_tree = EquationTree(equation_node)
    try:
        instantiated_equation = instantiate_constants(equation_tree, lambda: constant)
        # print(f"Instantiated equation: {instantiated_equation}")
        # evaluate the equation at 50 equally spaced points between -1 and 1        
        x_1 = np.linspace(-1, 1, 50)
        input_df = pd.DataFrame({"x_1": x_1.tolist()})
        # get f(x) values
        y = instantiated_equation.evaluate(input_df).astype(float)
        #y= instantiated_equation.get_evaluation(-1, 1, 50)
        if len(y) == 0:
            print(f"Failed to evaluate {equation} (y is empty)")
            return (None,)
        return input_df["x_1"].values.tolist(), y.tolist()
    except Exception as e:
        try: 
            x = symbols('x')
            x_1 = np.linspace(-1, 1, 50)
            input_df = pd.DataFrame({"x_1": x_1.tolist()})
            infix = infix.replace("X", "x").replace("c_0", str(constant))
            expr = sympify(infix)
            y = [expr.subs(x, x_val).evalf() for x_val in input_df["x_1"]]
            #print(y)
            return input_df["x_1"].values.tolist(), y
        except Exception as e:
            print(f"Failed to evaluate {equation}")
            print(e)
            return (None,)
    
def random_embedding(kind, model, dataset, units,assignment, classes=None, ): 
    """
    Generate a random embedding and plot the function graphs

    Parameters:
        kind: kind of model (AE, VAE, AE_C, VAE_C)
        model: model
        dataset: dataset
        units: number of latent dimensions
        assignment: assignment (is_function, is_operator, is_variable, is_constant)
        classes: classes
    
    Returns:
        number of functions that could be plotted
    """
    is_function, is_operator, is_variable, is_constant = assignment
    random_embedding = np.float32(np.random.random((100,units)))
    if kind != 'AE_C':
        random_equations, random_constants = decode_latent(model, dataset, random_embedding)
    else:
        random_equations, random_constants = decode_latent_classify(model, dataset, random_embedding, classes)
    final_random_equations = []
    final_random_constants = []
    final_random_embedding= []
    random_values = []
    for i, (e,c) in enumerate(zip(random_equations, random_constants)):
        c=c[0][0]
        values = generate_values(e,c, is_function, is_operator, is_variable, is_constant)
        if len(values) == 2:
            if type(values[1]) == list:
                is_complex = any([type(v) == complex for v in values[1]])
            else:
                is_complex = True
            if not is_complex:
                final_random_equations.append(e)
                final_random_constants.append(c)
                random_values.append(values)
                final_random_embedding.append(random_embedding[i])

    fig = plot_functions(equations=final_random_equations, constants=final_random_constants, values=random_values, is_function=is_function, is_operator=is_operator)
    if fig != None:
        print(f"Managed to plot {len(fig.data)} out of {len(random_equations)}")
        #correlation = correlation_coeff(torch.tensor(random_values)[:,1,:], torch.tensor(final_random_embedding))
        #print(f"The functions that could be plotted have a correlation_dis of {correlation[1]} and a correlation_cor of {correlation[0]}")
        return len(fig.data)
    else:
        return 0
    

def evaluate_different_models(d, batch_size, training_set_proportion, units, num_epochs, learning_rate, kind, weight, klweight = None, classes = None, assignments=None):
    """
    Function for the systematic evaluation of different models

    Parameters:
        d: dataset
        batch_size: batch size
        training_set_proportion: training set proportion
        units: number of latent dimensions
        num_epochs: number of epochs
        learning_rate: learning rate
        kind: kind of model (AE, VAE, AE_C, VAE_C)
        weight: weight for latent correlation loss
        klweight: kl weight
        classes: classes
        assignments: assignment (is_function, is_operator, is_variable, is_constant)
    
    Returns:
        results of the evaluation (dict)
    """
    torch.cuda.empty_cache()
    train_data, test_data, test_size = preprocessing(
        dataset=d,
        batch_size=batch_size,
        training_set_proportion=training_set_proportion
    )
    equations = [d.decode_equation(x[0]) for x in d]
    all_symbols = [item for sublist in equations for item in sublist]
    unique_symbols = sorted(list(set(all_symbols)))
    max_len = len(equations[0])
    #try:
    with io.capture_output() as captured:
        if kind == 'VAE':
            model, train_losses, test_losses, correlations_cor, correlations_dis, x_batches, x_hat_batches, df_results = training_VAE(train_data, test_data, units, unique_symbols, num_epochs, learning_rate, test_size, klweight)
        elif kind == 'AE':
            model, train_losses, test_losses, correlations_cor, correlations_dis, x_batches, x_hat_batches, df_results = training_AE(train_data, test_data, units, unique_symbols, num_epochs, learning_rate, test_size)
        elif kind == 'AE_C':
            model, train_losses, test_losses, correlations_cor, correlations_dis, correlations_dis_train, x_batches, x_hat_batches, df_results = training_AE_C(train_data, test_data, units, unique_symbols, num_epochs, learning_rate, test_size, max_len, classes, weight)
            x_hat_equations = [torch.argmax(batch[0], dim=1).tolist() for batch in x_hat_batches]
            x_hat_constants = [batch[:][1] for batch in x_hat_batches]
            for i, batch in enumerate(x_hat_equations):
                for j, eq in enumerate(batch):
                    x_hat_equations[i][j] = classes[eq]
            x_hat_batches = (x_hat_equations, x_hat_constants)
        elif kind == 'VAE_C':
            model, train_losses, test_losses, correlations_cor, correlations_dis, x_batches, x_hat_batches, df_results = training_VAE_C(train_data, test_data, units, unique_symbols, num_epochs, learning_rate, test_size, klweight, classes, max_len, weight)
            x_hat_equations = [torch.argmax(batch[0], dim=1).tolist() for batch in x_hat_batches]
            x_hat_constants = [batch[:][1] for batch in x_hat_batches]
            for i, batch in enumerate(x_hat_equations):
                for j, eq in enumerate(batch):
                    x_hat_equations[i][j] = classes[eq]
            x_hat_batches = (x_hat_equations, x_hat_constants)
        results, x_batches_p, x_hat_batches_p, x_constants_p, x_hat_constants_p = evaluation_ec(
            x_batches=x_batches,
            x_hat_batches=x_hat_batches, #(x_hat_equations, x_hat_constants),
            equation_tree_dataset=d,
            max_len=max_len,
            kind=kind,)
        dct = {
            'latent dims': units, 
            #'correlation_cor': float(df_results['correlation_cor']), 
            'correlation_dis' : float(df_results['correlation_dis']),
            #'correlation_dis reonstructed equations': distance_reconstructed_equations(model, d, test_data, kl_weight, is_vae)[0],
            #'valid reconstructed equations': distance_reconstructed_equations(model, d, test_data, kl_weight, is_vae)[1],
            'correlation_cor last 10 epochs': np.sum(correlations_cor[-10:]) / 10, 
            'correlation_dis last 10 epochs': np.sum(correlations_dis[-10:]) / 10, 
            'accuracy (individual)':results['accuracy (individual)'], 
            'accuracy equations': results['accuracy (equations)'], 
            'constant MSE': results['average mse constants'], 
            'average distance constants': results['average distance constants'],
            'learning_rate': learning_rate,
            'weight': weight,
            'kl_weight': klweight,
            
        }
        if kind == 'AE':
            dct['recovered equations'] = random_embedding(kind, model, d, units, assignments, classes),
        if kind == 'VAE':
            dct['recovered equations'] = random_embedding(kind, model, d, units, assignments, classes),
            #dct['weight'] = klweight
            dct['test_reconstruction_loss']= df_results['test_reconstruction_loss'][-1]
            dct['test_constant_loss']= df_results['test_constant_loss'][-1]
            dct['test_latent_correlation_loss']= df_results['test_latent_correlation_loss'][-1]
            dct['test_kl_divergence']= df_results['test_kl_divergence'][-1]
        if kind == 'AE_C':
            #dct['weight'] = weight
            dct['test_reconstruction_loss']= float(df_results['test_reconstruction_loss'][-1])
            dct['test_constant_loss']= float(df_results['test_constant_loss'][-1])
            dct['test_latent_correlation_loss']= float(df_results['test_latent_correlation_loss'][-1])
            dct['correlations_dis_train']= float(correlations_dis_train[-1])
        if kind == 'VAE_C':
            #dct['weight'] = klweight
            dct['test_reconstruction_loss']= float(df_results['test_reconstruction_loss'][-1])
            dct['test_constant_loss']= float(df_results['test_constant_loss'][-1])
            dct['test_latent_correlation_loss']= float(df_results['test_latent_correlation_loss'][-1])
            dct['test_kl_divergence'] = float(df_results['test_kl_divergence'][-1])
            #dct['correlations_dis_train']= float(correlations_dis_train[-1])

    return dct
            
    #except Exception as e:
    #    print(e)
    
def distance_reconstructed_equations(model, dataset, test_data, kl_weight, is_vae):
    """
    Calculate the distance between the reconstructed equations

    Parameters:
        model: model
        dataset: dataset
        test_data: test data
        kl_weight: kl weight
        is_vae: is vae
    """ 
    model.eval()
    embedding = []
    recon_equations = []
    recon_constants = []
    with torch.no_grad():
        for test_equations_batch, constant_list, values_batch in test_data:
            # constant_list = constant_list.unsqueeze(1).float()
            # Forward pass
            # recon_batch, mean, logvar = myvae(test_equations_batch)
            if not is_vae:
                (recon_batch_syntax, recon_batch_constants), z = model(
                    test_equations_batch, constant_list
                )
            else:
                (recon_batch_syntax, recon_batch_constants), z, mean, logvar = model(
                    test_equations_batch, constant_list
                )

            embedding += z.tolist()
            recon_equations += recon_batch_syntax
            recon_constants += recon_batch_constants
    # instantiate constant randomly
    y = []
    z = []
    recon_equations = np.array([x.detach().numpy() for x in recon_equations])
    recon_equations = np.argmax(recon_equations, axis=2)
    for i in range(len(recon_equations)):
        prefix_eq = dataset.decode_equation(recon_equations[i])
        result = generate_values(prefix_eq, recon_constants[i][0])
        if result != (None,):
            y.append(result[1])
            z.append(embedding[i])
    y = torch.tensor(y)
    z = torch.tensor(z)
    if len(y) > 0:
        correlation = correlation_coeff(y,z)[1]
    else:
        correlation = 0
    return float(correlation), len(y)
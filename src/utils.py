import matplotlib.pyplot as plt
from equation_tree.util.conversions import prefix_to_infix


global is_function, is_operator, is_variable

# is_function = lambda x: x in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "acos","asin",]
# is_operator = lambda x : x in ["+", "-", "*", "/", "**", "max", "min"]
# is_variable = lambda x : x in ["x_1"]

def plot_losses(train_loss, test_loss, correlation_cor=None, correlation_dis=None, df=None, save=None, correlations_dis_train=None, dpi=500):
    """
    Plot the train, test losses, correlation between latent space representation and value correlation

    Parameters:
    - train_loss (list): Train losses.
    - test_loss (list): Test losses.
    - correlation_cor (list): Correlation between latent space representation and value correlation.
    - correlation_dis (list): Correlation between latent space representation and value distance.
    - df (pd.DataFrame): Dataframe containing the individual losses.
    - save (str): Path to save the plot.
    - dpi (int): Dots per inch of the plot.

    """
    # make subplots
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(16, 10))
    fig.supxlabel('Epoch')
    fig.supylabel('Loss')

    ax[0,0].plot(train_loss, label='Train Loss')
    ax[0,0].plot(test_loss, label='Test Loss')
    ax[0,0].legend()
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_title('Train and Test Losses')
    #ax[0,0].suptitle('Train and Test Losses')
    if correlation_cor is not None:
        ax[1,0].plot(correlation_cor, label='Correlation_cor')
        ax[1,0].legend()
        ax[1,0].set_ylabel('Correlation')
        ax[1,0].set_title('Correlation between latent space representation and value correlation')
        #ax[1,0].suptitle('Correlation between latent space representation and value correlation')
    if correlation_dis is not None:
        ax[1,1].plot(correlation_dis, label='Correlation_dis')
        if correlations_dis_train is not None:
            ax[1,1].plot(correlations_dis_train, label='Correlation_dis_train')
        ax[1,1].legend()
        ax[1,1].set_ylabel('Correlation')
        ax[1,1].set_title('Correlation between latent space representation and value distance')
        #ax[1,1].suptitle('Correlation between latent space representation and value distance')
    if df is not None:
        ax[0,1].plot(df['test_reconstruction_loss'], label='Reconstruction Loss')
        if 'test_kl_divergence' in df.keys():
            ax[0,1].plot(df['test_kl_divergence'], label='KL Loss')
        ax[0,1].plot(df['test_constant_loss'], label='Constant Loss')
        
        ax[0,1].plot(df['test_latent_correlation_loss'], label='Distance Correlation Loss')
        average_loss = [x/4 for x in test_loss]
        #ax[0,1].plot(average_loss, label='Total Loss / 4')
        ax[0,1].legend()
        ax[0,1].set_ylabel('Loss')
        ax[0,1].set_title('Individual Test Losses')
        #ax[0,1].suptitle('Individual Test Losses')
    fig.show()
    if save is not None:
        fig.savefig(save, dpi=dpi)

def try_infix(equation, constant):
    """
    Try to convert the equation to infix notation and replace the constant.

    Parameters:
    - equation (str): Equation in prefix notation.
    - constant (float): Constant to replace.

    Returns:
    - str: Equation in infix notation with the constant replaced.
    """
    if type(constant) == list:
        constant = constant[0]
    try: 
        return prefix_to_infix(equation, is_function, is_operator).replace(
                            "c_1", str(round(constant, 2))
                        )
    except Exception as e:
        return str(equation)
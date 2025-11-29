import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def encode_onehot(X, n: int = None, negative_class: float = 0.0) -> np.ndarray:
    """
    Function for encoding ordinal labels into OneHot representation.
    
    Args:
        x (array_like): Input labels
        n (int): Set number of ordinal labels otherwise it's inferred from data
        negative_class (float): Default value for non-activated classes
    Returns:
        x_oh (np.ndarray): OneHot encoded labels
    """
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    x_oh = np.ones((len(X), n)) * negative_class
    for i in range(len(X)):
        m = X[i]
        x_oh[i,m] = 1
    return x_oh

def visualize_resuls(data_real: np.ndarray, data_gen: np.ndarray, labels: np.ndarray, labels_sampled: np.ndarray) -> None:
    # Find unique ordinal labels within samples
    labels_unique = np.unique(labels_sampled)

    # Sampling frequency of 5 min resolution in Hz
    fs = 1/(60*5)

    fig, ax = plt.subplots(len(labels_unique), 2, figsize=(8, 2*len(labels_unique)), layout='constrained')
    for i, label in enumerate(labels_unique):
        # Get indices to organize by ordinal labels
        indices_real, _ = np.nonzero(labels == label)
        indices_gen, _ = np.nonzero(labels_sampled == label)

        # Left column
        ax[i,0].fill_between(np.arange(0, data_real.shape[-1]), np.max(data_real[indices_real], 0), np.min(data_real[indices_real], 0), alpha=0.2)
        ax[i,0].plot(np.mean(data_real[indices_real], 0))
        ax[i,0].plot(data_gen[indices_gen].transpose(), alpha=0.5)
        ax[i,0].set_xlabel('Time (5 min)')
        ax[i,0].set_ylabel('Generated Power (MW)')
        ax[i,0].set_xlim(0, data_gen.shape[-1])
        ax[i,0].grid(True)

        # Right column
        f_real, pden_real = welch(data_real[indices_real,:], fs, nperseg=128)
        f_gen, pden_gen = welch(data_gen[indices_gen,:], fs, nperseg=128)
        ax[i,1].semilogy(f_real, np.mean(pden_real, 0))
        ax[i,1].fill_between(f_real, np.max(pden_real, 0), np.min(pden_real, 0), alpha=0.2)
        ax[i,1].semilogy(f_gen, pden_gen.transpose(), alpha=0.5)
        ax[i,1].set_xlim(0, max(f_real))
        ax[i,1].ticklabel_format(axis='x', style='sci', scilimits=(0, max(f_real)))
        ax[i,1].set_xlabel('Frequency (Hz)')
        ax[i,1].set_ylabel(r'PSD ($\frac{MW^2}{Hz}$)')
        ax[i,1].grid(True)

    # fig.savefig('generated_data.png', bbox_inches='tight')
    fig.show()
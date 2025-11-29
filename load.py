import csv
import numpy as np

def load_solar_data(path_data: str, path_labels: str) -> tuple:
    """
    Load and preprocess solar data and labels for GAN training.
    """
    with open(f'{path_data}' if path_data.endswith('csv') else f'{path_data}.csv', 'r') as csvfile:
        rows = np.array(list(csv.reader(csvfile)), dtype=float)
    # pecify according to time points in your own dataset
    rows = rows[:104832,:]
    print('\nShape rows (data raw):', rows.shape)

    with open(f'{path_labels}' if path_labels.endswith('csv') else f'{path_labels}.csv', 'r') as csvfile:
        labels = np.array(list(csv.reader(csvfile)), dtype=int)
    print('\nShape labels (labels raw):', labels.shape)

    # Transform data to conform with GAN image input dimensions (24x24 = 576)
    tr_x = np.reshape(rows.T,(-1,576))
    print('\nShape tr_x (data preprocessed):',tr_x.shape)
    tr_y = np.tile(labels,(32,1))
    print('\nShape tr_y (labels preprocessed):', tr_y.shape)
    m = np.ndarray.max(rows)
    print("\nMax(Solar) =", m)
    tr_x = tr_x/m

    return tr_x, tr_y, m

def load_wind_data(path_data: str, path_labels: str) -> tuple:
    """
    Load and preprocess solar data and labels for GAN training.
    """
    # Example dataset created for evnet_based GANs wind scenarios generation
    # Data from NREL wind integrated datasets
    with open(f'{path_data}' if path_data.endswith('csv') else f'{path_data}.csv', 'r') as csvfile:
        rows = np.array(list(csv.reader(csvfile)), dtype=float)
    tr_x = []
    m = np.ndarray.max(rows)
    print('\nMax(Wind)', m)
    
    for x in range(rows.shape[1]):
        train = rows[:-288, x].reshape(-1, 576)
        train = train / m
        tr_x.extend(train)

    tr_x = np.asarray(tr_x)
    print('\nShape tr_x', tr_x.shape)

    with open(f'{path_labels}' if path_labels.endswith('csv') else f'{path_labels}.csv', 'r') as csvfile:
        label = np.array(list(csv.reader(csvfile)), dtype=int)
    print('\nShape label', label.shape)
    
    return tr_x, label, m
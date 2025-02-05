import csv
import numpy as np



#Define function for loading solar data
def load_solar_data(path_data: str, path_labels:str) -> tuple:
    """
    Load and preprocess solar data and labels for GAN training
    """
    with open(f'{path_data}' if path_data.endswith('csv') else f'{path_data}.csv', 'r') as csvfile:
        rows = np.array([row for row in csv.reader(csvfile)], dtype=float)
    #Specify according to time points in your own dataset
    rows = rows[:104832,:]
    print('\nShape rows (data raw):', rows.shape)

    with open(f'{path_labels}' if path_labels.endswith('csv') else f'{path_labels}.csv', 'r') as csvfile:
        labels = np.array([row for row in csv.reader(csvfile)], dtype=int)
    print('\nShape labels (labels raw):', labels.shape)

    #Transform data to conform to GAN image input dimensions (24x24 = 576)
    trX = np.reshape(rows.T,(-1,576))
    print('\nShape trX (data preprocessed):',trX.shape)
    trY = np.tile(labels,(32,1))
    print('\nShape trY (labels preprocessed):', trY.shape)
    m = np.ndarray.max(rows)
    print("\nMax(Solar) =", m)
    trX = trX/m

    return trX, trY, m

#Define function for loading wind data
def load_wind_data(path_data: str, path_labels: str) -> tuple:
    """
    Load and preprocess solar data and labels for GAN training
    """
    #Example dataset created for evnet_based GANs wind scenarios generation
    # Data from NREL wind integrated datasets
    with open(f'{path_data}' if path_data.endswith('csv') else f'{path_data}.csv', 'r') as csvfile:
        rows = np.array([row for row in csv.reader(csvfile)], dtype=float)
    trX = []
    m = np.ndarray.max(rows)
    print('\nMax(Wind)', m)
    
    for x in range(rows.shape[1]):
        train = rows[:-288, x].reshape(-1, 576)
        train = train / m

        trX.extend(train)

    trX = np.asarray(trX)
    print('\nShape TrX', trX.shape)

    with open(f'{path_labels}' if path_labels.endswith('csv') else f'{path_labels}.csv', 'r') as csvfile:
        label = np.array([row for row in csv.reader(csvfile)], dtype=int)
    print('\nShape Label', label.shape)
    
    return trX, label, m

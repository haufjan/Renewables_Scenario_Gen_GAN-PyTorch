import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from load import load_solar_data, load_wind_data
from model import GAN
from util import visualize_resuls

def main(args: argparse.Namespace) -> tuple:
    # Check available device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print('Current device:', device)

    # Load data and labels
    tr_x = None
    tr_y = None
    m = None
    if args.data.endswith('solar.csv'):
        tr_x, tr_y, m = load_solar_data(args.data, args.label)
    elif args.data.endswith('wind.csv'):
        tr_x, tr_y, m = load_wind_data(args.data, args.label)
    
    # Determine number of unique labels
    events_num = len(np.unique(tr_y))

    # Instantiate model
    model = GAN(epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    dim_y=events_num,
                    device=device).to(device)
    
    # Start training
    model.fit(tr_x, tr_y)

    # Generate samples
    data_gen, labels_sampled = model.predict()

    # Rescaling
    data_real = tr_x*m
    data_gen = data_gen*m


    # Evaluation
    visualize_resuls(data_real, data_gen, tr_y, labels_sampled)
    plt.show()

    return data_gen, labels_sampled
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        help='Select data set for training',
        type=str
    )
    parser.add_argument(
        '--label',
        help='Select labels corresponding to data set',
        type=str
    )
    parser.add_argument(
        '--epochs',
        help='Training iterations',
        default=5000,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='Number of samples for one optimization',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate for optimizer',
        default=1e-4,
        type=float
    )

    args = parser.parse_args()

    data_gen, labels_sampled = main(args)
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs to train the model.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        default=32,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.25,
        help='Beta parameter for the Vector Quantizer loss.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='best_model.pth',
        help='Path to save the best model.'
    )
    parser.add_argument(
        '--encoder_depth',
        type=int,
        default=2,
        help='Depth of the Transformer encoder.'
    )
    parser.add_argument(
        '--decoder_depth',
        type=int,
        default=2,
        help='Depth of the Transformer decoder.'
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=4,
        help='Number of attention heads in the Transformer.'
    )
    parser.add_argument(
        '--num_embeddings',
        type=int,
        default=24,
        help='Number of embeddings for the Vector Quantizer.'
    )
    parser.add_argument(
        '--windows_door_in_condition',
        action='store_true',
        help = 'If set, the model will be trained with windows and doors in the condition.'
    )
    args = parser.parse_args()
    return args


DATA_DIR = '../3DFront/scenes'#rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"
OBJ_DIR = '../3DFront/objects'


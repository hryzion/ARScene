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
        '--bottleneck',
        type=str,
        default='ae',
        choices=['ae', 'vae', 'vqvae'],
    )
    parser.add_argument(
        '--windows_door_in_condition',
        action='store_true',
        help = 'If set, the model will be trained with windows and doors in the condition.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default = 'vqvae-s'
    )
    args = parser.parse_args()


    if args.model not in ['vqvae-s', 'vqvae-m', 'vqvae-l']:
        raise ValueError("Model must be one of 'vqvae-s', 'vqvae-m', or 'vqvae-l'.")
    
    if args.model == 'vqvae-s':
        args.encoder_depth = 2
        args.decoder_depth = 2
        args.heads = 4
        args.num_embeddings = 32
    elif args.model == 'vqvae-m':
        args.encoder_depth = 4
        args.decoder_depth = 4
        args.heads = 8
        args.num_embeddings = 64
    elif args.model == 'vqvae-l':
        args.encoder_depth = 6
        args.decoder_depth = 6
        args.heads = 16
        args.num_embeddings = 512

    if args.num_embeddings < 1:
        raise ValueError("Number of embeddings must be at least 1.")


    return args


DATA_DIR =  '/mnt/disk-1/zhx24/dataset/3dfront/Levels2021'
#'../3DFront/scenes'#rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"



OBJ_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'
#'../3DFront/objects'


"""InteriorGPT training with configurable room-condition dropout and epoch resume.

Set training.max_steps for an exact update budget and mid-epoch checkpoint resume.
Otherwise the original epoch-budget path below remains available.
YAML: model.cond_drop_rate (0..1, default 0.9), training.resume (auto/path/null),
training.seed (default 42), save.checkpoint_every (retained copies, in epochs).
super_parameters.epochs is the TOTAL budget, including previously completed epochs.

Examples:
    python train_ar.py --config CONFIG --output RUN
    python train_ar.py --config CONFIG --resume RUN/last.pt
    python train_ar.py --config CONFIG --output NEW_RUN --resume none --weights OLD.pth

last.pt saves model, Adam, epoch/step, metrics, config and RNG states after validation
at every epoch boundary. An interrupted partial epoch is replayed from that boundary.
sar_latest.pth remains a plain state_dict for existing inference scripts. Full resume
restores the saved optimizer learning rate; the current YAML controls condition dropout.
"""

import copy
from pathlib import Path

import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.quant import TokenSequentializer
from networks.roomlayout.RoomLayoutAutoRegressiveNet import RoomLayoutAutoRegressiveNet, resolve_cond_drop_rate
from networks.roomlayout.checkpoint import (
    CHECKPOINT_FORMAT, atomic_save, load_checkpoint, load_weights, loader_rng_state,
    resolve_resume, restore_loader_rng, restore_rng, rng_state, seed_rng, validate_resume_config,
)
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_arguments
import os
import wandb
from torchvision import models
from losses.ar_loss import MultiCodebookCrossEntropy
from utils import count_trainable_params

def train_model(
    model : RoomLayoutAutoRegressiveNet,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    save_path='./pretrained/interiorGPT/sar.pth',
    criterion = None,
    configs = None,
    use_wandb = False,
    resume = None,
    weights = None,
):  
    if (configs or {}).get('training', {}).get('max_steps') is not None:
        from networks.roomlayout.step_training import train_steps
        return train_steps(model, train_loader, val_loader, device, lr=lr, save_path=save_path,
                           criterion=criterion, configs=configs, use_wandb=use_wandb,
                           resume=resume, weights=weights)
    configs = copy.deepcopy(configs or {})
    training_config = configs.get('training', {})
    resume = training_config.get('resume') if resume is None else resume
    weights = training_config.get('weights') if weights is None else weights
    save_path = Path(save_path)
    output = save_path.parent
    resume_path = resolve_resume(resume, output)
    if resume_path is not None and weights is not None:
        raise ValueError('Use either resume or weights, not both')
    if resume_path is None and (output / 'last.pt').exists():
        raise ValueError('last.pt already exists; use --resume auto or choose a new --output')
    checkpoint_every = int(configs.get('save', {}).get('checkpoint_every', 200))
    if checkpoint_every < 0:
        raise ValueError('save.checkpoint_every must be nonnegative (0 disables retained copies)')

    t_sequentializer : TokenSequentializer = model.vae_token_sequentializer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')

    train_loss_recorder = []
    val_loss_recorder = []
    start_epoch = 0
    global_step = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        if checkpoint.get('training_mode') == 'steps':
            raise ValueError('This is a step checkpoint; set training.max_steps to resume it')
        validate_resume_config(checkpoint['config'], configs)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['next_epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        train_loss_recorder = checkpoint['train_losses']
        val_loss_recorder = checkpoint['val_losses']
        restore_loader_rng(train_loader, checkpoint['loader_rng']['train'])
        restore_loader_rng(val_loader, checkpoint['loader_rng']['val'])
        restore_rng(checkpoint['rng'])
        previous_rate = checkpoint['config'].get('model', {}).get('cond_drop_rate', 0.9)
        if previous_rate != model.cond_drop_rate:
            print(f'[Resume] condition drop rate: {previous_rate} -> {model.cond_drop_rate} (current YAML)')
        print(f'[Resume] {resume_path}: {start_epoch} epochs, {global_step} updates completed; '
              f'optimizer lr={optimizer.param_groups[0]["lr"]}')
        del checkpoint
    elif weights is not None:
        model.load_state_dict(load_weights(weights))
        print(f'[Weights] Loaded {weights}; starting epoch 1 with a fresh optimizer')

    configs.setdefault('model', {})['cond_drop_rate'] = model.cond_drop_rate
    print(f'[Training] room_mask_condition={model.room_mask_condition}, '
          f'cond_drop_rate={model.cond_drop_rate}, target_epochs={num_epochs}')
    if start_epoch >= num_epochs:
        print('[Training] The checkpoint already reached the requested total epochs')
        if use_wandb:
            wandb.finish()
        return train_loss_recorder, val_loss_recorder
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    with (output / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(configs, stream, allow_unicode=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            key_padding_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            
            with torch.no_grad():
                token_map = model.vae.encode_obj_tokens(obj_tokens, padding_mask=key_padding_mask)
                residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map)
                x_wo_first = t_sequentializer.generate_scaling_input(residual_fm_gt_list)
                residual_fm_gt = torch.cat(residual_fm_gt_list, dim = 1)
                encoded_gt = t_sequentializer.get_encoded_token_map(residual_fm_gt=residual_fm_gt)
            optimizer.zero_grad()

            logits= model(x_wo_first, text_desc, room_shape, encoded_gt)
            loss, loss_dict = criterion(logits, residual_fm_gt)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_recorder.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss =  {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device)
                key_padding_mask = batch['attention_mask'].to(device)
                text_desc = batch['text_desc']

                token_map = model.vae.encode_obj_tokens(obj_tokens, padding_mask=key_padding_mask)
                residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map)
                x_wo_first = t_sequentializer.generate_scaling_input(residual_fm_gt_list)
                residual_fm_gt = torch.cat(residual_fm_gt_list, dim = 1)
                encoded_gt = t_sequentializer.get_encoded_token_map(residual_fm_gt)

                logits= model(x_wo_first, text_desc, room_shape, encoded_gt)
                loss, loss_dict = criterion(logits, residual_fm_gt)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_recorder.append(avg_val_loss)
        best_val_loss = min(best_val_loss, avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} -  Val Loss =  {avg_val_loss:.4f}")

        if use_wandb:
            wandb.log({
                'train/total loss': avg_train_loss,
                'val/total': avg_val_loss,
                'epoch': epoch + 1,
                'global_step': global_step,
                'cond_drop_rate': model.cond_drop_rate,
            })

        checkpoint = {
            'format': CHECKPOINT_FORMAT,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'next_epoch': epoch + 1,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'train_losses': train_loss_recorder,
            'val_losses': val_loss_recorder,
            'config': configs,
            'rng': rng_state(),
            'loader_rng': {'train': loader_rng_state(train_loader), 'val': loader_rng_state(val_loader)},
        }
        atomic_save(checkpoint, output / 'last.pt')
        # Keep plain state_dict exports compatible with test_ar.py / intermediate_ar.py.
        atomic_save(checkpoint['model'], output / f'{save_path.stem}_latest.pth')
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            atomic_save(checkpoint, output / f'epoch_{epoch + 1:06d}.pt')
            atomic_save(checkpoint['model'], output / f'{save_path.stem}_{epoch}.pth')
    if use_wandb:
        wandb.finish()
    return train_loss_recorder, val_loss_recorder



if __name__ == "__main__":
    import yaml
    args = parse_arguments(ar_training=True)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['cond_drop_rate'] = resolve_cond_drop_rate(config)
    training_config = config.setdefault('training', {})
    if args.max_steps is not None:
        training_config['max_steps'] = args.max_steps
    resume = args.resume if args.resume is not None else training_config.get('resume')
    weights = args.weights if args.weights is not None else training_config.get('weights')
    training_config.update(resume=resume, weights=weights)
    seed_rng(int(training_config.get('seed', 42)))

    # --------------------- super_parameters -----------------------


    super_parameters = config['super_parameters']
    BATCH_SIZE = int(super_parameters.get('batch_size', 64))
    NUM_EPOCHS = int(super_parameters.get('epochs', 200))
    LEARNING_RATE = float(super_parameters.get('learning_rate', 1e-4))


    
    # --------------------- model -----------------------

    model_config = config['model']
    model_type = model_config.get("type")
    if model_type not in ['sar']:
        raise ValueError(f"Unsupported model type: {model_type}")

    sar_depth = int(model_config.get('depth', 8))
    sar_heads = int(model_config.get('heads', 8))
    mlp_ratio = float(model_config.get('mlp_ratio', 4.0))
    drop_rate = float(model_config.get('drop_rate', 0.1))
    attn_drop_rate = float(model_config.get('attn_drop_rate', 0.1))
    drop_path_rate = float(model_config.get('drop_path_rate', 0.1))
    shared_aln = model_config.get('shared_aln', False)
    norm_eps = float(model_config.get('norm_eps', 1e-6))
    attn_l2_norm = model_config.get('attn_l2_norm', False)
    use_prior_cluster = model_config.get('use_prior_cluster', False)

    word_dim = int(model_config.get('word_dim', 128))

    # --------------------- encoder -----------------------
    encoder_config = model_config.get('encoder')

    ENCODER_DEPTH = int(encoder_config.get('encoder_depth', 4))
    DECODER_DEPTH = int(encoder_config.get('decoder_depth', 4))
    num_bn = int(encoder_config.get('num_bottleneck', 8))
    num_recon = int(encoder_config.get('num_recon', 28))
    HEADS = int(encoder_config.get('num_heads', 4))
    NUM_EMBEDDINGS =  int(encoder_config.get('num_embeddings', 512))
    TOKEN_DIM  = int(encoder_config.get('token_dim', 64))

    encoder_pretrained_path = encoder_config.get('pretrained_path', None)



    # --------------------- dataset -----------------------
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/processed')
    dataset_padded_length = dataset_config.get('padded_length', None)
    dataset_filter = dataset_config.get('filter_fn',"all")
    dataset_num_class = dataset_config.get('num_class', 31)

    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

    # --------------------- save -----------------------

    save_config = config.setdefault('save', {})
    save_folder = save_config.get('save_folder', "./pretrained/interiorGPT")
    tag = f"{dataset_filter}"+"_depth"+str(sar_depth)+"_worddim"+str(word_dim)+f'_vocab{NUM_EMBEDDINGS}'
    if not config.get("text_condition", False):
        tag+='_wo_textc'
    if config.get('dataset', {}).get('use_objlat', True):
        tag+='_lat64'
    else:
        tag+='_wo_lat'
    if args.output:
        save_folder = args.output
    elif resume not in (None, 'auto', 'none'):
        save_folder = str(Path(resume).expanduser().parent)
    else:
        save_folder = save_config.get('run_directory', os.path.join(save_folder, tag))
    resume_path = resolve_resume(resume, save_folder)
    if resume_path is not None and weights is not None:
        raise ValueError('Use either resume or weights, not both')
    config['save']['save_folder'] = save_folder
    config['save']['run_directory'] = save_folder

    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")
    



    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')
    train_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='train',padded_length=dataset_padded_length, num_cate=dataset_num_class)
    val_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='test',padded_length=dataset_padded_length, num_cate=dataset_num_class)

    # Normalizer
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=dataset_num_class, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")


    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    ae_encoder = RoomLayoutVQVAE(token_dim=TOKEN_DIM, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, test_mode=True, configs=config,num_bottleneck=num_bn, num_recon=num_recon,num_classes=dataset_num_class).to(device)

    # ae_encoder = RoomLayoutVQVAE(token_dim=TOKEN_DIM, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, test_mode=True, configs=config).to(device)
    if resume_path is None and weights is None:
        ae_encoder.load_state_dict(torch.load(encoder_pretrained_path, map_location=device, weights_only=True))
    
    pretrained_resnet = models.ResNet18_Weights.IMAGENET1K_V1 if resume_path is None and weights is None else None
    feat_extractor = models.resnet18(weights=pretrained_resnet).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层

    
    sar = RoomLayoutAutoRegressiveNet(
        vae_local=ae_encoder, feature_extractor=feat_extractor,config=config, device=device,
        depth = sar_depth, embed_dim=word_dim, num_heads=sar_heads, mlp_ratio=mlp_ratio,
        drop_rate = drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        shared_aln=shared_aln, norm_eps=norm_eps, attn_l2_norm=attn_l2_norm,
        use_prior_cluster=use_prior_cluster
    ).to(device)

    ar_loss = MultiCodebookCrossEntropy(
        num_codebooks = sar.vae_token_sequentializer.num_codebooks, 
        vocab_size = sar.vae_token_sequentializer.vocab_size
    )
    total_p = count_trainable_params(sar)/1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")
    
    name = f"SceneGPT_{dataset_filter}_depth{sar_depth}_worddim{word_dim}"
    if not config.get("text_condition", False):
        name+='_wo_textc'
    if dataset_config.get('use_objlat'):
        name+='_lat64'
    if args.wandb:
        wandb.init(project="SceneGPT" ,name=name)

    train_model(
        model=sar,
        train_loader=train_loader,
        val_loader = val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_path=save_path,
        criterion=ar_loss,
        configs = config,
        use_wandb = args.wandb,
        resume=resume,
        weights=weights,
    )

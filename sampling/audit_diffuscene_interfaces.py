"""Read-only GPU audit of the trained path and legacy DiffuScene interfaces."""

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from networks.diffuscene.layout_data import LayoutDataset
from sample_layouts import decode_diffusion, write_json
from utils import THREED_FRONT_CATEGORY


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Submit the Slurm GPU wrapper; no CPU fallback')
    torch.set_num_threads(4)
    torch.manual_seed(219)
    run = Path('networks/diffuscene/runs/livingroom_uncond_100k')
    payload = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    config = payload['config']
    assert config == yaml.safe_load((run / 'config.yaml').read_text())
    net = config['network']
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=33, atiss=True, use_objlat=False)
    normalizer.load(run / 'normalizer_stats_atiss.json')
    for field, bounds in payload['normalizer'].items():
        for key, value in bounds.items():
            torch.testing.assert_close(normalizer.stats[field][key], value, rtol=0, atol=0)
    assert normalizer._metadata() == payload['normalizer_metadata']
    root = Path(config['dataset']['root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['categories'] == THREED_FRONT_CATEGORY
    assert manifest['geometry_convention'] == 'obb_center_full_size'

    data = LayoutDataset(root, 'train', limit=32)
    data.encode(normalizer, net['sample_num_points'])
    batch = {k: v.cuda() for k, v in default_collate([data[i] for i in range(len(data))]).items()}
    expected = torch.cat([batch['translations'], batch['sizes'], batch['angles'], batch['class_labels']], -1)
    assert expected.shape == (32, 27, 41)
    expected_independent = torch.zeros_like(expected)
    expected_independent[..., 6] = 1.
    expected_independent[..., 40] = 1.
    for i, row in enumerate(data.rows):
        n = len(row)
        for offset, field in ((0, 'translate'), (3, 'size')):
            lo, hi = (normalizer.stats[field][k].cuda() for k in ('min', 'max'))
            expected_independent[i, :n, offset:offset + 3] = (2 * (row[:, 33 + offset:36 + offset].cuda() - lo) / (hi - lo) - 1).clamp(-1, 1)
        expected_independent[i, :n, 6] = row[:, 39].cuda().cos()
        expected_independent[i, :n, 7] = row[:, 39].cuda().sin()
        expected_independent[i, :n, 8:] = row[:, :33].cuda()
    # CPU and CUDA trigonometric kernels may differ by a float32 ulp.
    torch.testing.assert_close(expected, expected_independent, rtol=1e-6, atol=1e-7)
    # A perfect normalized target must export exactly the original furniture set.
    decoded = decode_diffusion(expected.cpu(), normalizer, THREED_FRONT_CATEGORY)
    max_geometry_error = 0.
    for raw, result in zip(data.rows, decoded):
        assert result['status'] == 'complete' and len(result['objects']) == len(raw)
        for index, obj in enumerate(result['objects']):
            assert obj['category_id'] == int(raw[index, :33].argmax())
            geometry = torch.tensor(obj['position'] + obj['size'])
            torch.testing.assert_close(geometry, raw[index, 33:39], atol=2e-6, rtol=1e-6)
            max_geometry_error = max(max_geometry_error, float((geometry - raw[index, 33:39]).abs().max()))
            angle_error = np.arctan2(np.sin(obj['yaw'] - float(raw[index, 39])), np.cos(obj['yaw'] - float(raw[index, 39])))
            assert abs(angle_error) < 1e-6

    model = DiffusionSceneLayout_DDPM(33, None, net).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    report = {'slurm_job': os.environ.get('SLURM_JOB_ID'), 'checkpoint_step': payload['step'],
              'current_path': {'checkpoint_config_and_stats_match': True, 'category_mapping_match': True,
                               'real_scenes_checked': len(data), 'data_target_order_padding_and_raw_yaw': True,
                               'perfect_target_export_max_geometry_error_m': max_geometry_error},
              'legacy_interface_findings': {}}
    del payload
    with patch.object(model.diffusion, 'get_loss_iter', wraps=model.diffusion.get_loss_iter) as loss_call:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = model.get_loss(batch)
        assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in parts.values())
        torch.testing.assert_close(loss_call.call_args.args[0], expected, rtol=0, atol=0)
        train_condition = loss_call.call_args.kwargs['condition'].detach().clone()
        assert loss_call.call_args.kwargs['condition_cross'] is None
    report['current_path']['real_checkpoint_bf16_loss'] = float(loss)
    report['current_path']['train_condition_shape'] = list(train_condition.shape)
    assert train_condition.shape == (32, 27, 128)
    with patch.object(model.diffusion, 'gen_samples', return_value=expected) as sample_call:
        model.sample(batch_size=32, clip_denoised=False)
        torch.testing.assert_close(sample_call.call_args.kwargs['condition'], train_condition, rtol=0, atol=0)
        assert sample_call.call_args.kwargs['condition_cross'] is None
        assert sample_call.call_args.args[0] == torch.Size([32, 27, 41])
        assert sample_call.call_args.kwargs['clip_denoised'] is False
    report['current_path']['train_sample_condition_and_dimensions_match'] = True

    # Independently construct v targets and verify the loss and inverse for several t.
    diffusion = model.diffusion.diffusion
    x0 = expected[:4]
    timesteps = torch.tensor([0, 1, 500, 999], device='cuda')
    noise = torch.randn_like(x0)
    alpha = diffusion.alphas_cumprod.cuda()[timesteps].reshape(-1, 1, 1)
    v = alpha.sqrt() * noise - (1 - alpha).sqrt() * x0
    xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
    recovered = diffusion.model_predictions(lambda *args: v, xt, timesteps, None, None).pred_x_start
    torch.testing.assert_close(recovered, x0, atol=1e-6, rtol=1e-6)
    oracle_loss, _ = diffusion.p_losses(lambda *args: v, x0, timesteps, noise=noise)
    assert float(oracle_loss.max()) < 1e-12
    report['current_path']['v_target_and_reverse_formula_match'] = True

    findings = report['legacy_interface_findings']
    with patch.object(model.diffusion, 'gen_samples', return_value=expected) as ddpm, \
            patch.object(model.diffusion, 'gen_samples_ddim', return_value=expected) as ddim:
        model.sample(batch_size=32, ddim=True)
        findings['ddim_true_routing'] = {'ddpm_called': ddpm.called, 'ddim_called': ddim.called,
                                        'affects_current_samples': False}
    toy = torch.zeros(1, 2, 41, device='cuda')
    toy[..., 6] = 1.
    toy[0, 0, 8 + 1] = 1.  # one real object
    toy[0, 1, 8 + 32] = 1.  # one empty slot
    current = decode_diffusion(toy.cpu(), normalizer, THREED_FRONT_CATEGORY)
    assert len(current[0]['objects']) == 1
    try:
        old = model.delete_empty_from_network_samples(toy)
        findings['old_empty_filter'] = {'expected_objects': 1, 'observed_objects': old['sizes'].shape[1]}
    except Exception as exc:
        findings['old_empty_filter'] = {'error': str(exc)}
    for name, value, kwargs in [('old_keep_empty', toy, {'keep_empty': True}),
                                ('old_batch_two', toy.repeat(2, 1, 1), {'keep_empty': True})]:
        try:
            old = model.delete_empty_from_network_samples(value, **kwargs)
            findings[name] = {'sizes_shape': list(old['sizes'].shape)}
        except Exception as exc:
            findings[name] = {'error': str(exc)}
    # Exercise the invalid index on CPU to avoid poisoning the CUDA context.
    reserved = toy.cpu().clone()
    reserved[0, 0, 8 + 31] = 2.
    try:
        model.delete_empty_from_network_samples(reserved)
        findings['old_reserved_start'] = {'raises': False}
    except Exception as exc:
        findings['old_reserved_start'] = {'error': str(exc)}
    for name in ('old_empty_filter', 'old_keep_empty', 'old_batch_two', 'old_reserved_start'):
        findings[name]['affects_current_samples'] = False
    findings['old_dataset_adapter'] = {
        'room_shape_npz_expected_by_old_loader': True,
        'room_shape_npz_found_in_current_train': sum(1 for _ in (root / 'train').glob('*/room_shape.npz')),
        'current_loader_needs_this_file': False}
    findings['old_test_entrypoint'] = {'requires_super_parameters_missing_in_current_config': 'super_parameters' not in config,
                                        'expects_bare_state_dict_current_checkpoint_is_container': True,
                                        'affects_current_samples': False}

    # Verify ignored per-scene seeds with an actual short GPU diffusion chain.
    tiny = copy.deepcopy(net)
    tiny['sample_num_points'] = 4
    tiny['net_kwargs'].update(dim=32, dim_mults=[1, 1])
    tiny['diffusion_kwargs']['time_num'] = 8
    small = DiffusionSceneLayout_DDPM(33, None, tiny).cuda().eval()
    with torch.inference_mode():
        torch.manual_seed(191)
        first = small.sample(batch_size=2, batch_seeds=torch.tensor([11, 12]))
        torch.manual_seed(191)
        second = small.sample(batch_size=2, batch_seeds=torch.tensor([21, 22]))
        findings['batch_seeds_ignored'] = {'different_seeds_produce_identical_samples': torch.equal(first, second),
                                          'affects_current_samples': False,
                                          'current_sampler_sets_torch_manual_seed_per_batch': True}
    report['scope'] = 'Audit only. No training, weights, sampler settings, or saved samples changed.'
    destination = run / 'samples_last_uncond_1000_20260929/interface_audit.json'
    write_json(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

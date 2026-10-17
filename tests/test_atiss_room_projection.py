"""Room conditioning must reach the same tokens during training and generation."""

from pathlib import Path
import tempfile
import unittest

import torch

from datasets.atiss import ATISSDataset, fit_normalizer
from test_atiss_training import config_for, make_data
from train_atiss import build_model


class RoomProjectionTests(unittest.TestCase):
    def test_forward_decode_empty_prefix_and_state_reload(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            cfg = config_for(root)
            cfg['network'].update(hidden_dims=768, n_heads=8, query_dimensions=96, value_dimensions=96)
            data = ATISSDataset(root, 'train', num_classes=4, image_size=64)
            data.normalizer = fit_normalizer(data)
            item = next(data[(epoch, 0)] for epoch in range(100)
                        if int(data[(epoch, 0)]['lengths']) == 2)
            sample = data.collate_fn([item])
            for room_dim in (None, 64):
                with self.subTest(room_dim=room_dim):
                    if room_dim is not None:
                        cfg['network']['room_feature_dim'] = room_dim
                    torch.manual_seed(42)
                    model = build_model(cfg, pretrained=False).eval()
                    captured = []
                    handle = model.transformer_encoder.register_forward_pre_hook(
                        lambda module, args: captured.append(args[0].detach().clone()))
                    loss = model(sample).reconstruction_loss(sample, sample['lengths'])
                    boxes = {key: torch.cat([start, sample[key]], 1)
                             for key, start in model.start_symbol().items()}
                    with torch.no_grad():
                        model._encode(boxes, sample['room_layout'])
                        model._encode(model.start_symbol(), sample['room_layout'])
                    # Both inference branches and training see the same room/query tokens.
                    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
                    torch.testing.assert_close(captured[0][:, :2], captured[2], rtol=1e-6, atol=1e-6)
                    self.assertEqual(captured[0].shape[-1], 768)
                    with torch.no_grad():
                        raw = model.feature_extractor(sample['room_layout'])
                        room = model.fc_room_f(raw)
                        expected = model.fc(room) if room_dim is None else model.fc_room_token(room)
                    torch.testing.assert_close(captured[0][:, 0], expected, rtol=1e-6, atol=1e-6)
                    self.assertEqual(room.shape[-1], room_dim or 512)
                    loss.backward()
                    self.assertGreater(float(model.fc_room_f.weight.grad.norm()), 0)
                    self.assertGreater(float(model.fc.weight.grad.norm()), 0)
                    self.assertTrue(all(p.grad is None for p in model.feature_extractor.parameters()))
                    if room_dim is not None:
                        self.assertGreater(float(model.fc_room_token.weight.grad.norm()), 0)
                    else:
                        self.assertFalse(any(k.startswith('fc_room_token.') for k in model.state_dict()))
                    handle.remove()
                    restored = build_model(cfg, pretrained=False).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    with torch.no_grad():
                        restored_loss = restored(sample).reconstruction_loss(sample, sample['lengths'])
                    torch.testing.assert_close(loss.detach(), restored_loss, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

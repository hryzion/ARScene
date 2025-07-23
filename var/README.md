This folder contains the VAR model for quantization and token generation.

## Usage

```python
from VAR import build_vae_var, RoomEncoder

vae, var = build_vae_var(...params)  # for vae and var model
room_encoder = RoomEncoder(...params) # for room structure and prompt encoder
```

At train time, use the forward pass of VAE and VAR. At inference time the forward pass should never be used(because it uses teacher forcing); use `VAR.autoregressive_infer_cfg` and the util methods in VQVAE instead. Refer to var.py and vqvae.py for more information.

When tokenizing prompt data, ensure a `[CLS]` tag and a `[SEP]` tag is attached to both ends respectively, in accordance to BERT.  

## Todo

- Presently the Encoder and Decoder in basic_vae.py are dummy placeholders. Replace them with valid Encoder/Decoder or update the import in vqvae.py. 
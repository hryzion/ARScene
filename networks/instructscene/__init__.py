from typing import *
from torch import Tensor
from torch.optim import Optimizer

from functools import partial

from torch import optim

from .objfeat_vqvae import ObjectFeatureVQVAE
from .sg_diffusion_vq_objfeat import SgObjfeatVQDiffusion
from .sg2sc_diffusion import Sg2ScDiffusion


_DIFFUSION_MODEL_KEYS = {
    "attn_dim",
    "t_dim",
    "edge_attn_dim",
    "n_heads",
    "n_layers",
    "dropout",
    "gated_ff",
    "ada_norm",
    "use_e_cross_attn",
    "concat_global_condition",
    "use_positional_encoding",
    "num_objfeat_embeds",
    "objfeat_dim",
    "objfeat_pool_layers",
    "objfeat_out_layers",
}


def _normalize_model_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    if "depth" in out and "n_layers" not in out:
        out["n_layers"] = out.pop("depth")
    if "heads" in out and "n_heads" not in out:
        out["n_heads"] = out.pop("heads")
    return out


def diffusion_kwargs_from_config(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    model_cfg = config.get("model", config)
    merged = {}
    merged.update(model_cfg.get("common", {}))
    merged.update({
        k: v for k, v in model_cfg.items()
        if k in _DIFFUSION_MODEL_KEYS or k in ("depth", "heads")
    })
    merged.update(model_cfg.get(section, {}))
    merged = _normalize_model_keys(merged)
    return {k: v for k, v in merged.items() if k in _DIFFUSION_MODEL_KEYS}


def model_size_name_from_config(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", config)
    explicit = model_cfg.get("size_name") or model_cfg.get("model_size")
    if explicit:
        return str(explicit)
    sg_cfg = diffusion_kwargs_from_config(config, "sg")
    sg2sc_cfg = diffusion_kwargs_from_config(config, "sg2sc")
    sg_depth = sg_cfg.get("n_layers", 5)
    sg_attn = sg_cfg.get("attn_dim", 512)
    sg_heads = sg_cfg.get("n_heads", 8)
    sg2sc_depth = sg2sc_cfg.get("n_layers", 5)
    sg2sc_attn = sg2sc_cfg.get("attn_dim", 512)
    sg2sc_heads = sg2sc_cfg.get("n_heads", 8)
    if (sg_depth, sg_attn, sg_heads) == (sg2sc_depth, sg2sc_attn, sg2sc_heads):
        return f"d{sg_depth}_a{sg_attn}_h{sg_heads}"
    return f"sg_d{sg_depth}_a{sg_attn}_h{sg_heads}_sg2sc_d{sg2sc_depth}_a{sg2sc_attn}_h{sg2sc_heads}"


def append_model_size_to_folder(folder: str, config: Dict[str, Any]) -> str:
    import os
    size_name = model_size_name_from_config(config)
    normalized = os.path.normpath(folder)
    if os.path.basename(normalized) == size_name:
        return folder
    return os.path.join(folder, size_name)


def model_from_config(
    config: Dict[str, Any],
    num_objs=-1, num_preds=-1,
    text_emb_dim=512,
    **kwargs
):
    # Scene graph to scene layout
    if "sg2sc" in config["name"]:
        if "diffusion" in config["name"]:
            return Sg2ScDiffusion(
                num_objs, num_preds,
                use_objfeat="objfeat" in config["name"],
                **diffusion_kwargs_from_config({"model": config}, "sg2sc")
            )
        else:
            raise NotImplementedError(f"Unknown model name: {config['name']}")

    # Instruction to scene graph
    elif "sg" in config["name"]:
        if "vq_objfeat" in config["name"]:
            return SgObjfeatVQDiffusion(
                num_objs, num_preds,
                text_emb_dim=text_emb_dim,
                **diffusion_kwargs_from_config({"model": config}, "sg")
            )
        else:
            raise NotImplementedError(f"Unknown model name: {config['name']}")

    # Object feature VQ VAE
    elif "objfeatvqvae" in config["name"]:
        return ObjectFeatureVQVAE(
            config["objfeat_type"],
            config["vq_type"],
            **kwargs
        )

    else:
        raise NotImplementedError(f"Unknown model name: {config['name']}")


def optimizer_from_config(config: Dict[str, Any], params: Iterable[Tensor]) -> Optimizer:
    name = config["name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.)

    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay
    )

    # SGD config
    momentum = config.get("momentum", 0.)
    nesterov = config.get("nesterov", False)

    # Adam/AdamW/Radam config
    betas = config.get("betas", (0.9, 0.999))

    optimizer = {
        "sgd": partial(optim.SGD, momentum=momentum, nesterov=nesterov, **kwargs),
        "adam": partial(optim.Adam, betas=betas, **kwargs),
        "adamw": partial(optim.AdamW, betas=betas, **kwargs),
        "radam": partial(optim.RAdam, betas=betas, **kwargs),
    }[name]

    return optimizer(params)

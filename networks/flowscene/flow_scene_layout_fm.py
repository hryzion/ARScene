import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel

from ..diffuscene.denoise_net import Unet1D
from ..stats_logger import StatsLogger


class FlowSceneLayout_FM(Module):
    def __init__(self, n_classes, feature_extractor, config):
        super().__init__()

        self.room_mask_condition = config.get("room_mask_condition", True)
        self.text_condition = config.get("text_condition", False)
        self.text_glove_embedding = config.get("text_glove_embedding", False)
        self.text_clip_embedding = config.get("text_clip_embedding", False)
        if self.room_mask_condition:
            self.feature_extractor = feature_extractor
            self.fc_room_f = nn.Linear(512, config["latent_dim"])
        elif self.text_condition:
            text_embed_dim = config.get("text_embed_dim", 512)
            if self.text_glove_embedding:
                self.fc_text_f = nn.Linear(50, text_embed_dim)
            elif self.text_clip_embedding:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, _ = clip.load("ViT-B/32", device=device)
                for p in self.clip_model.parameters():
                    p.requires_grad = False
            else:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
                self.bertmodel = BertModel.from_pretrained("bert-base-cased")
                for p in self.bertmodel.parameters():
                    p.requires_grad = False
                self.fc_text_f = nn.Linear(768, text_embed_dim)

        if config["net_type"] == "unet1d":
            self.velocity_net = Unet1D(**config["net_kwargs"])
        else:
            raise NotImplementedError()

        self.n_classes = n_classes
        self.config = config

        self.objectness_dim = config.get("objectness_dim", 1)
        self.class_dim = config.get("class_dim", 21)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)

        self.learnable_embedding = config.get("learnable_embedding", False)
        self.instance_condition = config.get("instance_condition", False)
        self.sample_num_points = config.get("sample_num_points", 12)
        self.instance_emb_dim = config.get("instance_emb_dim", 64)

        if self.learnable_embedding:
            if self.instance_condition:
                self.register_parameter(
                    "positional_embedding",
                    nn.Parameter(torch.randn(self.sample_num_points, self.instance_emb_dim)),
                )
            else:
                self.instance_emb_dim = 0
        else:
            if self.instance_condition:
                self.fc_instance_condition = nn.Sequential(
                    nn.Linear(self.sample_num_points, self.instance_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.instance_emb_dim, self.instance_emb_dim, bias=False),
                )
            else:
                self.instance_emb_dim = 0

        self.room_partial_condition = config.get("room_partial_condition", False)
        self.partial_num_points = config.get("partial_num_points", 0)
        self.partial_emb_dim = config.get("partial_emb_dim", 64)
        if self.room_partial_condition:
            self.fc_partial_condition = nn.Sequential(
                nn.Linear(
                    self.bbox_dim + self.class_dim + self.objectness_dim + self.objfeat_dim,
                    self.partial_emb_dim,
                    bias=False,
                ),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(self.partial_emb_dim, self.partial_emb_dim, bias=False),
            )
        else:
            self.partial_emb_dim = 0

        self.room_arrange_condition = config.get("room_arrange_condition", False)
        self.arrange_emb_dim = config.get("arrange_emb_dim", 64)
        if self.room_arrange_condition:
            self.fc_arrange_condition = nn.Sequential(
                nn.Linear(
                    self.size_dim + self.class_dim + self.objectness_dim + self.objfeat_dim,
                    self.arrange_emb_dim,
                    bias=False,
                ),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(self.arrange_emb_dim, self.arrange_emb_dim, bias=False),
            )
        else:
            self.arrange_emb_dim = 0

        flow_kwargs = config.get("flow_kwargs", {})
        diffusion_kwargs = config.get("diffusion_kwargs", {})
        self.time_scale = float(flow_kwargs.get("time_scale", diffusion_kwargs.get("time_num", 1000)))
        self.num_steps = int(flow_kwargs.get("num_steps", 100))
        self.max_grad_norm = float(flow_kwargs.get("max_grad_norm", 10.0))
        self.clip_denoised = bool(flow_kwargs.get("clip_denoised", False))

    def _build_room_layout_target(self, sample_params):
        if self.objectness_dim > 0:
            objectness = sample_params["objectness"]
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        if self.objfeat_dim > 0:
            if self.objfeat_dim == 32:
                objfeats = sample_params["objfeats_32"]
            else:
                objfeats = sample_params["objfeats"]

        if self.room_arrange_condition and self.config["point_dim"] == self.translation_dim + self.angle_dim:
            return torch.cat([translations, angles], dim=-1).contiguous()

        if self.config["point_dim"] == self.bbox_dim + self.class_dim + self.objectness_dim + self.objfeat_dim:
            if self.objectness_dim > 0:
                room_layout_target = torch.cat(
                    [translations, sizes, angles, class_labels, objectness], dim=-1
                ).contiguous()
            else:
                room_layout_target = torch.cat([translations, sizes, angles, class_labels], dim=-1).contiguous()
            if self.objfeat_dim > 0:
                room_layout_target = torch.cat([room_layout_target, objfeats], dim=-1).contiguous()
        elif self.config["point_dim"] == self.bbox_dim:
            room_layout_target = torch.cat([translations, sizes, angles], dim=-1).contiguous()
        else:
            raise NotImplementedError

        return room_layout_target

    def _build_condition(self, room_layout, batch_size, num_points, device, sample_params=None, text=None, partial_boxes=None, input_boxes=None):
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_layout))
        else:
            room_layout_f = None

        if self.instance_condition:
            if self.learnable_embedding:
                instance_indices = torch.arange(self.sample_num_points).long().to(device)[None, :].repeat(batch_size, 1)
                instan_condition_f = self.positional_embedding[instance_indices, :]
            else:
                instance_label = torch.eye(self.sample_num_points).float().to(device)[None, ...].repeat(batch_size, 1, 1)
                instan_condition_f = self.fc_instance_condition(instance_label)
        else:
            instan_condition_f = None

        if room_layout_f is not None and instan_condition_f is not None:
            condition = torch.cat(
                [room_layout_f[:, None, :].repeat(1, num_points, 1), instan_condition_f], dim=-1
            ).contiguous()
        elif room_layout_f is not None:
            condition = room_layout_f[:, None, :].repeat(1, num_points, 1)
        elif instan_condition_f is not None:
            condition = instan_condition_f
        else:
            condition = None

        if self.room_partial_condition:
            if sample_params is None:
                if partial_boxes is None:
                    raise ValueError("partial_boxes must be provided for partial condition during sampling")
                zeros_boxes = torch.zeros((batch_size, num_points - partial_boxes.shape[1], partial_boxes.shape[2])).float().to(device)
                partial_input = torch.cat([partial_boxes, zeros_boxes], dim=1).contiguous()
            else:
                room_layout_target = self._build_room_layout_target(sample_params)
                partial_valid = torch.ones((batch_size, self.partial_num_points, 1)).float().to(device)
                partial_invalid = torch.zeros((batch_size, num_points - self.partial_num_points, 1)).float().to(device)
                partial_mask = torch.cat([partial_valid, partial_invalid], dim=1).contiguous()
                partial_input = room_layout_target * partial_mask
            partial_condition_f = self.fc_partial_condition(partial_input)
            condition = torch.cat([condition, partial_condition_f], dim=-1).contiguous() if condition is not None else partial_condition_f

        if self.room_arrange_condition:
            if sample_params is None:
                if input_boxes is None:
                    raise ValueError("input_boxes must be provided for arrange condition during sampling")
                arrange_input = torch.cat(
                    [input_boxes[:, :, self.translation_dim : self.translation_dim + self.size_dim], input_boxes[:, :, self.bbox_dim :]],
                    dim=-1,
                ).contiguous()
            else:
                sizes = sample_params["sizes"]
                class_labels = sample_params["class_labels"]
                if self.objectness_dim > 0:
                    objectness = sample_params["objectness"]
                if self.objfeat_dim > 0:
                    if self.objfeat_dim == 32:
                        objfeats = sample_params["objfeats_32"]
                    else:
                        objfeats = sample_params["objfeats"]

                parts = [sizes, class_labels]
                if self.objectness_dim > 0:
                    parts.append(objectness)
                if self.objfeat_dim > 0:
                    parts.append(objfeats)
                arrange_input = torch.cat(parts, dim=-1).contiguous()
            arrange_condition_f = self.fc_arrange_condition(arrange_input)
            condition = torch.cat([condition, arrange_condition_f], dim=-1).contiguous() if condition is not None else arrange_condition_f

        if self.text_condition:
            if self.text_glove_embedding:
                if sample_params is not None:
                    condition_cross = self.fc_text_f(sample_params["desc_emb"])
                else:
                    condition_cross = self.fc_text_f(text)
            elif self.text_clip_embedding:
                if sample_params is not None:
                    tokenized = clip.tokenize(sample_params["description"]).to(device)
                else:
                    tokenized = clip.tokenize(text).to(device)
                condition_cross = self.clip_model.encode_text(tokenized)
            else:
                if sample_params is not None:
                    tokenized = self.tokenizer(sample_params["description"], return_tensors="pt", padding=True).to(device)
                else:
                    tokenized = self.tokenizer(text, return_tensors="pt", padding=True).to(device)
                text_f = self.bertmodel(**tokenized).last_hidden_state
                condition_cross = self.fc_text_f(text_f)
        else:
            condition_cross = None

        return condition, condition_cross

    def get_loss(self, sample_params):
        class_labels = sample_params["class_labels"]
        room_layout = sample_params["room_layout"]
        batch_size, num_points, _ = class_labels.shape
        device = class_labels.device

        room_layout_target = self._build_room_layout_target(sample_params)
        condition, condition_cross = self._build_condition(
            room_layout=room_layout,
            batch_size=batch_size,
            num_points=num_points,
            device=device,
            sample_params=sample_params,
        )

        x0 = room_layout_target
        x1 = torch.randn_like(x0)
        t = torch.rand((batch_size,), device=device).clamp(1e-5, 1.0 - 1e-5)
        t_view = t.view(batch_size, 1, 1)
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_v = x1 - x0
        beta = t * self.time_scale
        pred_v = self.velocity_net(xt, beta, context=condition, context_cross=condition_cross)

        mse = (pred_v - target_v) ** 2
        loss = mse.mean()

        loss_dict = {"loss.fm": loss.detach()}

        if x0.shape[-1] == self.translation_dim + self.angle_dim:
            loss_trans = mse[:, :, 0 : self.translation_dim].mean()
            loss_angle = mse[:, :, self.translation_dim : self.translation_dim + self.angle_dim].mean()
            loss_dict.update({"loss.trans": loss_trans.detach(), "loss.angle": loss_angle.detach()})
            return loss, loss_dict

        if x0.shape[-1] == self.objectness_dim + self.class_dim + self.bbox_dim + self.objfeat_dim:
            loss_trans = mse[:, :, 0 : self.translation_dim].mean()
            loss_size = mse[:, :, self.translation_dim : self.translation_dim + self.size_dim].mean()
            loss_angle = mse[:, :, self.translation_dim + self.size_dim : self.bbox_dim].mean()
            loss_bbox = mse[:, :, 0 : self.bbox_dim].mean()
            loss_class = mse[:, :, self.bbox_dim : self.bbox_dim + self.class_dim].mean()
            loss_dict.update(
                {
                    "loss.trans": loss_trans.detach(),
                    "loss.size": loss_size.detach(),
                    "loss.angle": loss_angle.detach(),
                    "loss.bbox": loss_bbox.detach(),
                    "loss.class": loss_class.detach(),
                }
            )

            if self.objectness_dim > 0:
                start = self.bbox_dim + self.class_dim
                end = start + self.objectness_dim
                loss_object = mse[:, :, start:end].mean()
                loss_dict["loss.object"] = loss_object.detach()

            if self.objfeat_dim > 0:
                start = self.bbox_dim + self.class_dim + self.objectness_dim
                loss_objfeat = mse[:, :, start:].mean()
                loss_dict["loss.objfeat"] = loss_objfeat.detach()

        return loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        room_mask,
        num_points,
        point_dim,
        batch_size=1,
        text=None,
        partial_boxes=None,
        input_boxes=None,
        ret_traj=False,
        clip_denoised=None,
        num_steps=None,
        batch_seeds=None,
    ):
        device = room_mask.device

        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        if num_steps is None:
            num_steps = self.num_steps

        condition, condition_cross = self._build_condition(
            room_layout=room_mask,
            batch_size=batch_size,
            num_points=num_points,
            device=device,
            sample_params=None,
            text=text,
            partial_boxes=partial_boxes,
            input_boxes=input_boxes,
        )

        x = torch.randn((batch_size, num_points, point_dim), device=device)
        dt = 1.0 / float(num_steps)
        traj = [x] if ret_traj else None

        if partial_boxes is not None:
            zeros_boxes = torch.zeros((batch_size, num_points - partial_boxes.shape[1], partial_boxes.shape[2])).float().to(device)
            partial_padded = torch.cat([partial_boxes, zeros_boxes], dim=1).contiguous()
            partial_mask = torch.zeros((batch_size, num_points, 1), device=device)
            partial_mask[:, : partial_boxes.shape[1], :] = 1.0

        for i in range(num_steps, 0, -1):
            t = torch.full((batch_size,), float(i) / float(num_steps), device=device)
            beta = t * self.time_scale
            v = self.velocity_net(x, beta, context=condition, context_cross=condition_cross)
            x = x - v * dt
            if clip_denoised:
                x = torch.clamp(x, -1.0, 1.0)
            if partial_boxes is not None:
                x = x * (1.0 - partial_mask) + partial_padded * partial_mask
            if ret_traj:
                traj.append(x)

        return traj if ret_traj else x

    @torch.no_grad()
    def generate_layout(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False):
        samples = self.sample(
            room_mask=room_mask,
            num_points=num_points,
            point_dim=point_dim,
            batch_size=batch_size,
            text=text,
            ret_traj=ret_traj,
            clip_denoised=clip_denoised,
            batch_seeds=batch_seeds,
        )
        return self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)

    @torch.no_grad()
    def delete_empty_from_network_samples(self, samples, device="cpu", keep_empty=False):
        samples_dict = {
            "translations": samples[:, :, 0 : self.translation_dim].contiguous(),
            "sizes": samples[:, :, self.translation_dim : self.translation_dim + self.size_dim].contiguous(),
            "angles": samples[:, :, self.translation_dim + self.size_dim : self.bbox_dim].contiguous(),
            "class_labels": nn.functional.one_hot(
                torch.argmax(samples[:, :, self.bbox_dim : self.bbox_dim + self.class_dim - 1].contiguous(), dim=-1),
                num_classes=self.n_classes - 2,
            ),
            "objectness": samples[:, :, self.bbox_dim + self.class_dim - 1 : self.bbox_dim + self.class_dim] >= 0,
        }
        if self.objfeat_dim > 0:
            samples_dict["objfeats"] = samples[:, :, self.bbox_dim + self.class_dim : self.bbox_dim + self.class_dim + self.objfeat_dim]

        boxes = {
            "objectness": torch.zeros(1, 0, 1, device=device),
            "class_labels": torch.zeros(1, 0, self.n_classes - 2, device=device),
            "translations": torch.zeros(1, 0, self.translation_dim, device=device),
            "sizes": torch.zeros(1, 0, self.size_dim, device=device),
            "angles": torch.zeros(1, 0, self.angle_dim, device=device),
        }
        if self.objfeat_dim > 0:
            boxes["objfeats"] = torch.zeros(1, 0, self.objfeat_dim, device=device)

        max_boxes = samples.shape[1]
        for i in range(max_boxes):
            if not keep_empty and samples_dict["objectness"][0, i, -1] > 0:
                continue
            for k in samples_dict.keys():
                if k == "class_labels":
                    boxes[k] = torch.cat(
                        [boxes[k], samples[:, i : i + 1, self.bbox_dim : self.bbox_dim + self.class_dim - 1].to(device)], dim=1
                    )
                    boxes["objectness"] = torch.cat(
                        [boxes["objectness"], samples[:, i : i + 1, self.bbox_dim + self.class_dim - 1 : self.bbox_dim + self.class_dim].to(device)],
                        dim=1,
                    )
                else:
                    boxes[k] = torch.cat([boxes[k], samples_dict[k][:, i : i + 1, :].to(device)], dim=1)

        if self.objfeat_dim > 0:
            return {
                "class_labels": boxes["class_labels"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu"),
                "objfeats": boxes["objfeats"].to("cpu"),
            }
        return {
            "class_labels": boxes["class_labels"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu"),
        }


def train_on_batch(model, optimizer, sample_params, config):
    optimizer.zero_grad()
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = float(v.item()) if torch.is_tensor(v) else float(v)
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
    StatsLogger.instance()["gradnorm"].value = grad_norm.item()
    StatsLogger.instance()["lr"].value = optimizer.param_groups[0]["lr"]
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = float(v.item()) if torch.is_tensor(v) else float(v)
    return loss.item()

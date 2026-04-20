from src.net_builder import get_net_builder
from src.peft import get_peft_config
net_name = "timm/vit_base_patch16_clip_224.openai"
net_from_name = False
peft_config = get_peft_config({
    "method_name": "lora",       # any truthy presence triggers petl_builder freeze rules
    "lora_bottleneck": 4,
    "freeze_backbone": False,   # optional; see _DEFAULT_PEFT_CONFIG in peft.py
})
vit_config = {"drop_path_rate": 0.0}
net_builder = get_net_builder(
    net_name,
    net_from_name,
    peft_config=peft_config,
    vit_config=vit_config,
)
model = net_builder(
    num_classes=5,
    pretrained=True,
    pretrained_path="",  # optional: path to checkpoint dict with 'model' key
)
print("Loaded model")
# Forward (TimmViTWrapper): dict with "logits" and "feat"
import torch
x = torch.randn(2, 3, 224, 224)
out = model(x)
logits = out["logits"]   # [2, 5]
feat = out["feat"]       # [2, embed_dim]
print("Forwarded model")
print("Logits: ", logits.shape)
print("Feat: ", feat.shape)
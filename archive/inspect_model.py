import torch
from custom_model_v3_phase2 import DeBERTaV3CrossAttention

device = torch.device("cpu")
model = DeBERTaV3CrossAttention()
ckpt = torch.load("checkpoints/best_v3_p2_restaurant.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()

print("VA Head Bias:", model.va_head[-1].bias.detach().numpy())
print("VA Head Weights (mean):", model.va_head[-1].weight.mean().item())

# Sample some layers
print("Log Var V:", model.log_var_v.item())
print("Log Var A:", model.log_var_a.item())

print("Arousal Norm Weight:", model.arousal_norm.norm.weight.detach().numpy())
print("Arousal Norm Bias:", model.arousal_norm.norm.bias.detach().numpy())

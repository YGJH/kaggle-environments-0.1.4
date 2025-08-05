#!/usr/bin/env python3
import torch

# 載入模型檢查權重鍵名
checkpoint = torch.load("checkpoints/best_model_wr_0.750.pt", map_location="cpu")
state_dict = checkpoint['model_state_dict']

print("模型權重鍵名:")
for key in state_dict.keys():
    print(f"  {key}: {state_dict[key].shape}")

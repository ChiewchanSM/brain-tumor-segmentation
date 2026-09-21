import torch
from model   import build_resnet50_segmentation_model
from dataset import build_dataset, CLASS_NAMES
from train   import run_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = build_resnet50_segmentation_model(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load("outputs/best_model_XXXX.pt", map_location=device))

# Load dataset (only need test split)
_, _, test_loader = build_dataset("D:/ComputerVision/FinalProject/Dataset", batch_size=8)

# Evaluate
results = run_epoch(model, test_loader, device, epoch=1, total_epochs=1, phase="test")
print("\n── Test Results ──")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")
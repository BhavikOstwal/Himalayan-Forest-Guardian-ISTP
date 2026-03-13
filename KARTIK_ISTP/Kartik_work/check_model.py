import torch

cp = torch.load('output/cnn_model_20260130_180918/best_model.pt', map_location='cpu')
print('Epoch:', cp['epoch'])
print('Val F1:', cp['val_f1'])
print('\nModel state dict keys:')
for k, v in cp['model_state_dict'].items():
    print(f'  {k}: {v.shape}')

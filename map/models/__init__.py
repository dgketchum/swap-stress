import torch

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VG_PARAMS = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
DROP_FEATURES = ['MGRS_TILE', 'lat', 'lon']
BATCH_SIZE = 512

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

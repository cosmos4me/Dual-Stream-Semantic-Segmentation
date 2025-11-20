LEARNING_RATE = 3.0e-4
BATCH_SIZE = 4
EPOCHS = 40 
TARGET_SIZE = 512
NUM_CLASSES = 2

CHECKPOINT = "/content/drive/MyDrive/v8.pth" # You should change your own path
AIR_ENCODER_CHECKPOINT = "/content/drive/MyDrive/Air_encoder_ConvNeXt.pth"
VISION_ENCODER_CHECKPOINT = "/content/drive/MyDrive/M3_final_seg.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TR_IMG_DIR = "/content/ts_sn"; TR_LAB_DIR = "/content/tl_sn"
TR_AP_DIR = "/content/t_ap"; TR_GEMS_DIR = "/content/t_gems"
VL_IMG_DIR = "/content/vs_sn"; VL_LAB_DIR = "/content/vl_sn"
VL_AP_DIR = "/content/v_ap"; VL_GEMS_DIR = "/content/v_gems"

TRAIN_SUBSET_RATIO = 1.0
VAL_SUBSET_RATIO = 1.0

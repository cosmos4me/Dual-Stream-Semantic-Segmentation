import torch
import torch.nn as nn
import numpy as np
import itertools
import torchmetrics
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 우리가 만든 모듈들
from config import *
from dataset import DualStreamDataset, collate_fn
from model import DualStream_DFormer_Model

# ==========================================
# 1. Helper Functions (Freeze/Unfreeze)
# ==========================================
def freeze_encoders(model):
    """V8 모델의 Vision/Air 인코더를 동결합니다."""
    print("--- [Train Strategy] 1단계: 인코더(Vision, Air) 동결 ---")
    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    for param in model.air_encoder.parameters():
        param.requires_grad = False
    print("--- 인코더가 동결되었습니다. 디코더와 융합 모듈만 학습합니다. ---")

def unfreeze_encoders(model):
    """V8 모델의 모든 파라미터를 해제합니다."""
    print("--- [Train Strategy] 2단계: 모델 전체 동결 해제 ---")
    for param in model.parameters():
        param.requires_grad = True
    print("--- 모델 전체가 파인튜닝을 시작합니다. ---")

# ==========================================
# 2. Training & Evaluation Functions
# ==========================================
def train_one_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Train (AMP)", leave=False)
    
    for batch in progress_bar:
        if batch is None: continue
        pixel_values = batch['pixel_values'].to(device)
        air_values = batch['air_values'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=pixel_values, air_values=air_values, labels=labels)
            loss = outputs['loss']
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, val_loader, metrics_dict, device):
    model.eval()
    total_loss = 0
    # 메트릭 초기화
    for metric in metrics_dict.values(): 
        metric.reset()
        
    progress_bar = tqdm(val_loader, desc="Validate", leave=False)
    for batch in progress_bar:
        if batch is None: continue
        pixel_values = batch['pixel_values'].to(device)
        air_values = batch['air_values'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=pixel_values, air_values=air_values, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits'] # (B, 512, 512)

        total_loss += loss.item()
        
        # 메트릭 업데이트
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        for metric in metrics_dict.values(): 
            metric.update(preds, labels)

    avg_loss = total_loss / len(val_loader)
    computed_metrics = {name: metric.compute().item() for name, metric in metrics_dict.items()}
    return avg_loss, computed_metrics

# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    print(f"사용 중인 디바이스: {DEVICE}")
    print(f"Vision Target Size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Air/GEMS Target Size: 64x64 (고정)")

    # --- 데이터셋 로드 ---
    print("\n--- 데이터셋 로드 (Config 경로 사용) ---")
    try:
        full_train_dataset = DualStreamDataset(TR_IMG_DIR, TR_LAB_DIR, TR_AIR_DIR, TR_GEMS_DIR, TARGET_SIZE)
        full_val_dataset = DualStreamDataset(VL_IMG_DIR, VL_LAB_DIR, VL_AIR_DIR, VL_GEMS_DIR, TARGET_SIZE)
        
        train_size = int(len(full_train_dataset) * TRAIN_SUBSET_RATIO)
        val_size = int(len(full_val_dataset) * VAL_SUBSET_RATIO)
        
        np.random.seed(42)
        train_indices = np.random.choice(len(full_train_dataset), train_size, replace=False)
        val_indices = np.random.choice(len(full_val_dataset), val_size, replace=False)
        
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)

        print(f"훈련 샘플: {len(train_dataset)}개 / 검증 샘플: {len(val_dataset)}개")
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    except Exception as e:
        print(f"데이터셋 로드 중 치명적 오류: {e}")
        return

    # --- 모델 빌드 ---
    print("\n--- DFormer 모델 빌드 (차등 학습률 적용) ---")
    model = DualStream_DFormer_Model(
        num_classes=NUM_CLASSES,
        vis_channels=4,
        air_channels_input=48,
        air_channels_list=[64, 128, 320, 512],
        segformer_id="nvidia/segformer-b1-finetuned-ade-512-512",
        focal_weight=FOCAL_WEIGHT,
        lov_weight=LOVASZ_WEIGHT,
        final_loss_weight=FINAL_LOSS_WEIGHT,
        aux_f2_loss_weight=AUX_F2_LOSS_WEIGHT,
        aux_f3_loss_weight=AUX_F3_LOSS_WEIGHT
    )

    # --- 가중치 로드 (Air Encoder) ---
    print(f"--- Air Encoder 가중치 로드 시도: {AIR_ENCODER_CHECKPOINT} ---")
    try:
        air_state_dict = torch.load(AIR_ENCODER_CHECKPOINT, map_location=DEVICE)
        model.air_encoder.load_state_dict(air_state_dict, strict=True)
        print("--- Air Encoder 가중치 로드 성공 ---")
    except Exception as e:
        print(f"--- [오류] Air Encoder 로드 실패: {e} ---")

    # --- 가중치 로드 (Visual Encoder) ---
    print(f"--- Visual Encoder 가중치 로드 시도: {VISION_ENCODER_CHECKPOINT} ---")
    try:
        vis_state_dict = torch.load(VISION_ENCODER_CHECKPOINT, map_location=DEVICE)
        
        # Prefix 정리 로직
        prefix_to_remove_1 = "_orig_mod."
        if any(k.startswith(prefix_to_remove_1) for k in vis_state_dict.keys()):
            print(f"    (!!!) '{prefix_to_remove_1}' 접두사 감지. 1차 정리합니다.")
            vis_state_dict = {k[len(prefix_to_remove_1):]: v for k, v in vis_state_dict.items() if k.startswith(prefix_to_remove_1)}
            
        prefix_to_remove_2 = "segformer."
        if any(k.startswith(prefix_to_remove_2) for k in vis_state_dict.keys()):
            print(f"    (!!!) '{prefix_to_remove_2}' 접두사 감지. 2차 정리합니다.")
            vis_state_dict = {k[len(prefix_to_remove_2):]: v for k, v in vis_state_dict.items() if k.startswith(prefix_to_remove_2)}
        else:
            print("    (참고) 2차 정리 대상인 'segformer.' 접두사가 없습니다.")

        missing_keys, unexpected_keys = model.visual_encoder.load_state_dict(vis_state_dict, strict=True)
        print("--- Visual Encoder 가중치 로드 성공 ---")
        if missing_keys: print(f"    (로드 안 됨 - 오류 발생): {missing_keys}")
        if unexpected_keys: print(f"    (불필요한 키 - 오류 발생): {unexpected_keys}")
    except Exception as e:
        print(f"--- [오류] Visual Encoder 가중치 로드 실패: {e} ---")

    model.to(DEVICE)

    # --- 옵티마이저 설정 (차등 학습률) ---
    VISUAL_LR = LEARNING_RATE / 10
    AIR_LR = LEARNING_RATE / 2
    fusion_params = itertools.chain(
        model.film_f1.parameters(),
        model.gate_f1.parameters(),
        model.film_f2.parameters(),
        model.gate_f2.parameters(),
        model.ca_f3.parameters(),
        model.ca_f4.parameters()
    )

    param_groups = [
        {"params": model.visual_encoder.parameters(), "lr": VISUAL_LR},
        {"params": model.air_encoder.parameters(), "lr": AIR_LR},
        {"params": fusion_params, "lr": LEARNING_RATE},
        {"params": model.decoder.parameters(), "lr": LEARNING_RATE}
    ]
    
    optimizer = AdamW(param_groups, weight_decay=1e-5)
    print(f"차등 학습률 적용:")
    print(f"  Visual Encoder (LR={VISUAL_LR:.1e})")
    print(f"  Air Encoder (LR={AIR_LR:.1e})")
    print(f"  Fusion/Decoder (LR={LEARNING_RATE:.1e})")

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    print(f"--- 스케줄러 변경: CosineAnnealingLR (T_max={EPOCHS}, eta_min=1e-6) ---")

    # --- 메트릭 설정 (Torchmetrics) ---
    metrics_collection = torchmetrics.MetricCollection({
        'IoU': torchmetrics.JaccardIndex(task="binary", num_classes=NUM_CLASSES, ignore_index=255),
        'Acc': torchmetrics.Accuracy(task="binary", num_classes=NUM_CLASSES, ignore_index=255)
    }).to(DEVICE)

    # --- 훈련 루프 시작 ---
    print(f"\n--- DFormer (v4 Deep Supervision) 융합 모델 검증 훈련 시작 ({EPOCHS} 에포크) ---")
    scaler = torch.amp.GradScaler('cuda')
    best_iou = -1.0

    # 1단계: 인코더 동결 후 학습
    freeze_encoders(model)
    print(f"동결후 2회 초기 훈련 ON")
    for i in range(2):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
        val_loss, val_metrics = evaluate(model, val_loader, metrics_collection, DEVICE)
        val_iou = val_metrics['IoU']
        val_acc = val_metrics['Acc']
        
        print(f" [Freeze Epoch {i+1}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val IoU: {val_iou:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
            print(f"  (***) 최고 IoU 갱신: {best_iou:.4f}")

    # 2단계: 전체 동결 해제 후 학습
    unfreeze_encoders(model)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE)
        val_loss, val_metrics = evaluate(model, val_loader, metrics_collection, DEVICE)
        val_iou = val_metrics['IoU']
        val_acc = val_metrics['Acc']

        print(f"Epoch {epoch} 완료:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val IoU: {val_iou:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")

        scheduler.step()
        # optimizer.param_groups[0]은 visual encoder lr입니다.
        print(f"  Vis LR: {optimizer.param_groups[0]['lr']:.2e} | Main LR: {optimizer.param_groups[2]['lr']:.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
            print(f"  (***) 최고 IoU 갱신: {best_iou:.4f}")

    print("\n--- DFormer 검증 훈련 종료 ---")
    print(f"최고 검증 IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
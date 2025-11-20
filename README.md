# DualStream-DFormer: Multi-Modal Semantic Segmentation

**DualStream-DFormer**는 고해상도 위성 이미지(Vision)와 대기 오염 데이터(Air Quality)를 융합하여 정밀한 영역 분할(Semantic Segmentation)을 수행하는 하이브리드 딥러닝 모델입니다.

서로 다른 해상도와 특성을 가진 이종 데이터를 효과적으로 결합하기 위해 **SegFormer**와 **ConvNeXt**를 기반으로 한 듀얼 스트림 구조를 채택하였으며, **FiLM**, **Gating**, **Cross-Attention** 메커니즘을 통해 특징을 단계별로 융합합니다.

## 🌟 Key Features

* **Dual-Stream Architecture**:
    * **Vision Stream**: `SegFormer-B1` (Pretrained)을 사용하여 시각적 특징 추출
    * **Air Quality Stream**: Custom `ConvNeXt` Encoder를 사용하여 48채널 대기 데이터 처리
* **Advanced Feature Fusion**:
    * **Low-Level**: `FiLM (Feature-wise Linear Modulation)` + `Dynamic Fusion Gate`
    * **High-Level**: `Cross-Attention Block` (Global Context Modeling)
* **Robust Decoder Design**:
    * **ASPP (Atrous Spatial Pyramid Pooling)**: 멀티 스케일 문맥 포착
    * **Attention Gates**: Skip Connection 정보의 선택적 융합
    * **ConvNeXt Refinement**: 디코딩 단계에서의 세밀한 특징 복원
* **Loss Function Strategy**:
    * **Hybrid Loss**: Focal Loss + Lovasz-Softmax Loss (Class Imbalance 해결)
    * **Deep Supervision**: 중간 레이어(Auxiliary Heads)에서도 손실을 계산하여 학습 안정성 확보

## 🏗️ Model Architecture

| Stage | Interaction Method | Purpose |
| :--- | :--- | :--- |
| **Encoder** | SegFormer (Vis) + ConvNeXt (Air) | 각각의 모달리티에서 계층적 특징 추출 |
| **Stage 1-2** | **FiLM + Gating** | 채널별 특징 변조 및 지역적 정보 융합 |
| **Stage 3-4** | **Cross-Attention** | 전역적 문맥 정보 교환 및 상호 연관성 학습 |
| **Decoder** | **ASPP + Attention Gate** | 경계면 정제 및 해상도 복원 |

## 📂 Directory Structure

데이터셋은 `DualStreamDataset` 클래스에 맞춰 다음과 같은 구조로 구성되어야 합니다.

```plaintext
/content/
├── ts_sn/      # Train Images (Satellite)
├── tl_sn/      # Train Labels (Masks)
├── t_ap/       # Train Air Pollution Data (TIF)
├── t_gems/     # Train GEMS Data (TIF)
├── vs_sn/      # Validation Images
├── vl_sn/      # Validation Labels
├── v_ap/       # Validation Air Pollution Data
└── v_gems/     # Validation GEMS Data
```

## ⚙️ Configuration

주요 하이퍼파라미터는 코드 최상단에서 설정 가능합니다.

```python
LEARNING_RATE = 3.0e-4
BATCH_SIZE = 4
EPOCHS = 40
TARGET_SIZE = 512
NUM_CLASSES = 2 (Background / Target)

# Loss Weights
FOCAL_WEIGHT = 0.5
LOVASZ_WEIGHT = 0.5
AUX_LOSS_WEIGHTS = {'final': 1.0, 'f2': 0.3, 'f3': 0.15}
```

## 🚀 Usage

### 1. Requirements

필요한 라이브러리를 설치합니다.

```bash
pip install torch torchvision rasterio opencv-python transformers tqdm
```

### 2. Training

모델 학습을 시작하는 기본 코드 예시입니다.

```python
from torch.utils.data import DataLoader
import torch.optim as optim

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 및 로더
train_dataset = DualStreamDataset(TR_IMG_DIR, TR_LAB_DIR, TR_AP_DIR, TR_GEMS_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 모델 초기화
model = DualStream_DFormer_Model(num_classes=1).to(device)

# 옵티마이저 및 스케일러
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

# 학습 루프
model.train()
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
    print(f"Epoch {epoch+1} Loss: {loss:.4f}")
    
    # 체크포인트 저장
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"{CHECKPOINT}_ep{epoch+1}.pth")
```

### 3. Inference

학습된 모델을 사용하여 추론을 수행합니다.

```python
model.eval()
with torch.no_grad():
    output = model(pixel_values=test_img, air_values=test_air)
    logits = output['logits']
    prediction = (torch.sigmoid(logits) > 0.5).long()
```

## 🧩 Technical Details

### Loss Functions
* **Focal Loss**: 데이터 불균형이 심한 세그멘테이션 작업에서 어려운 샘플(Hard negatives)에 더 큰 가중치를 부여합니다.
* **Lovasz Loss**: IoU(Intersection over Union) 지표를 직접 최적화하도록 설계된 손실 함수입니다.

### Data Preprocessing
* **Vision**: 0~1 사이로 정규화 (Rescaling), 512x512 리사이즈.
* **Air Quality**: GEMS 및 오염물질 데이터를 채널 방향으로 결합(Concatenate) 후 정규화, 64x64로 리사이즈.

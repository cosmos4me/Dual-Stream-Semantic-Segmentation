# 🛰️ DualStream-DFormer

**DualStream-DFormer** is a PyTorch-based deep learning model designed for **Multi-Modal Semantic Segmentation**. It effectively fuses **Satellite Imagery (Vision)** and **Air Quality Data (Auxiliary)** to improve segmentation performance on complex ground targets.

## 🌟 Key Features

* **Hybrid Dual-Encoder Architecture:**
    * **Vision Stream:** Uses **SegFormer (B1)** adapted for 4-channel satellite input. 

[Image of Transformer Encoder Architecture]

    * **Air Quality Stream:** Uses a custom **ConvNeXt-based** encoder for 48-channel environmental data.
* **Advanced Fusion Modules:**
    * **Low-Level:** FiLM (Feature-wise Linear Modulation) + Dynamic Gated Fusion.
    * **High-Level:** Cross-Attention mechanisms for semantic feature alignment.
* **Robust Training Strategy:**
    * **2-Stage Training:** Automatically handles **Frozen Encoder** warm-up followed by **Full Fine-tuning**.
    * **Differential Learning Rates:** Applies different learning rates to Encoders, Fusion modules, and Decoders.
* **Loss Function:** Optimizes using a weighted combination of **Focal Loss** and **Lovasz Softmax Loss**.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `config.py` | **Configuration entry point.** Contains paths, hyperparameters, and device settings. |
| `model.py` | Implementation of the DualStream architecture (Encoders, Fusion, Decoder). |
| `dataset.py` | Custom `DualStreamDataset` for loading satellite and air quality GeoTIFFs. |
| `losses.py` | Custom loss implementations (`FocalLoss`, `LovaszLoss`, `DiceLoss`). |
| `train.py` | Main training script handling the 2-stage strategy and validation. |

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies. It is recommended to use a virtual environment (e.g., Conda).

```bash
pip install -r requirements.txt

### 2. Data Preparation
Organize your dataset directories. The model expects Satellite Images, Labels, Air Pollution Maps, and GEMS Data.

Recommended structure:

/data_root/
├── ts_sn/   # Train Satellite Images
├── tl_sn/   # Train Labels
├── t_ap/    # Train Air Pollution (.tif)
├── t_gems/  # Train GEMS (.tif)
├── vs_sn/   # Validation Satellite Images
├── vl_sn/   # Validation Labels
└── ...

3. Configuration
Crucial Step: Open config.py and modify the paths to match your environment.

Update Paths: Change TR_IMG_DIR, VL_IMG_DIR, CHECKPOINT_SAVE_PATH, etc.

Hyperparameters: Adjust BATCH_SIZE, EPOCHS, and LEARNING_RATE if needed.

4. Training
Run the main training script. The code automatically manages the 2-stage training process (freezing encoders initially, then unfreezing).



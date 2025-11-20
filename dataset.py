import os
import cv2
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
 
class DualStreamDataset(Dataset):
    def __init__(self, image_dir, label_dir, air_dir, gems_dir, target_size=512):
        try:
            self.base_filenames = sorted([f for f in os.listdir(image_dir) if not f.startswith('.') and os.path.isfile(os.path.join(image_dir, f))])
            if len(self.base_filenames) == 0: raise FileNotFoundError(f"{image_dir} 에서 파일을 찾을 수 없습니다.")
        except FileNotFoundError as e:
            print(f"오류: {e}")
            raise e
            
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.air_dir = air_dir
        self.gems_dir = gems_dir
        self.target_size = target_size
        self.gems_prefix = "GEMS_"
        self.air_prefixes = ["AIR_Pollution_CO_", "AIR_Pollution_NO2_", "AIR_Pollution_SO2_"]
        print(f"데이터셋 로드: {image_dir}, 총 {len(self.base_filenames)}개 샘플.")

    def _process_image(self, path):
        with rasterio.open(path) as src: image = src.read()
        image_rescaled = (image / 256).astype(np.uint8)
        image_hwc = np.transpose(image_rescaled, (1, 2, 0))
        resized_image = cv2.resize(image_hwc, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        image_chw = np.transpose(resized_image, (2, 0, 1))
        return torch.from_numpy(image_chw).float() / 255.0

    def _process_aux_tif(self, path, num_channels=12):
        aux_target_size = 64
        with rasterio.open(path) as src: aux_map = src.read()
        if aux_map.shape[0] != num_channels: aux_map = aux_map[:num_channels, :, :]
        aux_map_hwc = np.transpose(aux_map, (1, 2, 0))
        resized_aux_map = cv2.resize(aux_map_hwc, (aux_target_size, aux_target_size), interpolation=cv2.INTER_LINEAR)
        if resized_aux_map.ndim == 2: resized_aux_map = np.expand_dims(resized_aux_map, axis=2)
        resized_aux_map_chw = np.transpose(resized_aux_map, (2, 0, 1))
        min_val, max_val = np.min(resized_aux_map_chw), np.max(resized_aux_map_chw)
        if (max_val - min_val) > 1e-6:
            aux_map_normalized = (resized_aux_map_chw - min_val) / (max_val - min_val)
        else:
            aux_map_normalized = np.zeros_like(resized_aux_map_chw)
        return aux_map_normalized

    def _process_label(self, path):
        with rasterio.open(path) as src: label = src.read(1)
        resized_label = cv2.resize(label, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        new_label = np.zeros_like(resized_label, dtype=np.int64)
        new_label[resized_label == 90] = 1
        return torch.from_numpy(new_label).long()

    def __getitem__(self, idx):
        base_name = self.base_filenames[idx]
        try:
            img_path = os.path.join(self.image_dir, base_name)
            label_path = os.path.join(self.label_dir, base_name)
            aux_maps = []
            gems_path = os.path.join(self.gems_dir, self.gems_prefix + base_name)
            aux_maps.append(self._process_aux_tif(gems_path)) # (12, 64, 64)
            for prefix in self.air_prefixes:
                air_path = os.path.join(self.air_dir, prefix + base_name)
                aux_maps.append(self._process_aux_tif(air_path)) # 3 * (12, 64, 64)
            combined_aux_map = np.concatenate(aux_maps, axis=0) # (48, 64, 64)
            return {
                "pixel_values": self._process_image(img_path),
                "air_values": torch.from_numpy(combined_aux_map).float(),
                "labels": self._process_label(label_path)
            }
        except Exception as e:
            print(f"오류 (샘플 {idx}, {base_name}): {e}")
            return None

    def __len__(self):
        return len(self.base_filenames)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

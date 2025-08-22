# %%
import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_comparisons(root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for category in ["upper", "lower"]:
        category_dir = os.path.join(root_dir, category)
        if not os.path.exists(category_dir):
            continue
        
        for sample_id in os.listdir(category_dir):
            sample_dir = os.path.join(category_dir, sample_id)
            gt_dir = os.path.join(sample_dir, "gt_mask")
            pred_dir = os.path.join(sample_dir, "pred_mask")
            if not (os.path.exists(gt_dir) and os.path.exists(pred_dir)):
                continue
            
            gt_files = sorted(os.listdir(gt_dir))
            pred_files = sorted(os.listdir(pred_dir))
            
            # 匹配相同视角
            paired_files = [(gt, pred) for gt, pred in zip(gt_files, pred_files) if os.path.splitext(gt)[0] == os.path.splitext(pred)[0]]
            if not paired_files:
                continue
            
            fig, axes = plt.subplots(len(paired_files), 2, figsize=(6, 3 * len(paired_files)))
            if len(paired_files) == 1:
                axes = [axes]  # 单行处理
            
            for idx, (gt_file, pred_file) in enumerate(paired_files):
                gt_img = Image.open(os.path.join(gt_dir, gt_file)).convert("RGB")
                pred_img = Image.open(os.path.join(pred_dir, pred_file)).convert("RGB")
                
                axes[idx][0].imshow(gt_img)
                axes[idx][0].set_title("Ground Truth")
                axes[idx][0].axis("off")
                
                axes[idx][1].imshow(pred_img)
                axes[idx][1].set_title("Prediction")
                axes[idx][1].axis("off")
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{category}_{sample_id}.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

# 使用示例
root_folder = "exp/baseline_reproduce/predict_masks"
save_folder = "exp/baseline_reproduce/comparison_results"
visualize_comparisons(root_folder, save_folder)

# %%
import numpy as np
import torch
coord = np.load('tmp/coords.npy')
coord = torch.from_numpy(coord).float()
grid_size = 2.0
grid_coord = torch.div(coord - coord.min(0)[0], grid_size, rounding_mode="trunc").int()

# 统计每个体素里的点数
coords = grid_coord
coords_np = coords.cpu().numpy()

import numpy as np
from collections import Counter

coord_tuples = [tuple(c) for c in coords_np]
count_dict = Counter(coord_tuples)

print("总点数:", len(coord_tuples))
print("唯一voxel数:", len(count_dict))
print("平均每个voxel点数:", np.mean(list(count_dict.values())))
print("最大voxel点数:", np.max(list(count_dict.values())))
print("最小voxel点数:", np.min(list(count_dict.values())))
# %%
import matplotlib.pyplot as plt
# voxel 内点数列表
voxel_counts = list(count_dict.values())

# 绘制直方图
plt.figure(figsize=(8,5))
plt.hist(voxel_counts, bins=range(1, max(voxel_counts)+2), color='skyblue', edgecolor='black', align='left')
plt.xlabel("Number of points per voxel")
plt.ylabel("Number of voxels")
plt.title("Voxel Point Count Distribution")
plt.xticks(range(1, max(voxel_counts)+1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# %%

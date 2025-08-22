import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.PTv1.point_transformer_seg import PointTransformerSeg38
from models.PTv3.point_transformer_v3 import PointTransformerSeg
from models.Seg2d.pspnet import PSPNet
from utils.color_utils import color2label


class ToothSegNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=17, 
                 grid_size = 0.05,
                 pretrain_2d_path='.checkpoints/PSPNet/train_ade20k_pspnet50_epoch_100.pth',
                 use_pretrain=None,
                 ):
        super().__init__()

        self.use_pretrain = use_pretrain
        self.grid_size = grid_size

        self.seg_model_3d = PointTransformerSeg(
            in_channels=in_channels, num_classes=num_classes, enable_pic_feat=True)
 
        self.seg_model_2d = PSPNet(layers=50, classes=num_classes+1, zoom_factor=8, use_ppm=True, pretrained=True, output_intermediate=False)

        # 加载训练权重
        if use_pretrain is not None:
            
            print(f"Loading pretrained model from {use_pretrain}")
            pretrained_model = torch.load(use_pretrain)
            self.load_state_dict(pretrained_model['model_state_dict'], strict=True)
        
        else:
            # 加载 2D 预训练模型
            print(f"Loading 2d pretrained model from {pretrain_2d_path}")
            pretrained_model = torch.load(pretrain_2d_path)
            pretrained_weights = pretrained_model['state_dict']
            pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
            # 过滤掉和分类头相关的权重（cls 和 aux）
            keys_to_remove = [k for k in pretrained_weights.keys() if k.startswith('cls.4') or k.startswith('aux.4')]
            for k in keys_to_remove:
                pretrained_weights.pop(k)

            self.seg_model_2d.load_state_dict(pretrained_weights, strict=False)


    def project_points(self, cameras_Rt, cameras_K, points, render_size, 
                       normalize=False):
        """
        cameras_Rt: (B, N_v, 4, 4)
        cameras_K:  (B, N_v, 3, 3)
        points:     (B, N_pc, 3)
        render_size: (H, W)

        Returns:
            projected points: (B, N_v, N_pc, 2)
        """
        B, N_v, _, _ = cameras_Rt.shape
        _, N_pc, _ = points.shape

        # 1. 变成齐次坐标 (B, N_pc, 4)
        ones = torch.ones((B, N_pc, 1), device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=-1)  # (B, N_pc, 4)

        # 2. 外参变换：points_cam = Rt @ points_homo
        Rt = cameras_Rt[:, :, :3, :]
        point_cam = torch.einsum('bvrc,bnc->bvnr', Rt, points_homo)  # (B, N_v, 3, 4) * (B, N_pc, 4) -> (B, N_v, N_pc, 3)

        # 3. 内参投影：K @ points_cam
        point_img = torch.einsum('bvrc,bvnc->bvnr', cameras_K, point_cam)  # (B, N_v, 3, 3) * (B, N_v, N_pc, 3) -> (B, N_v, N_pc, 3)

        # 5. 归一化：除以z
        img_x = point_img[..., 0] / (point_img[..., 2] + 1e-6) # x is w
        img_y = point_img[..., 1] / (point_img[..., 2] + 1e-6) # y is h


        img_points = torch.stack([img_y, render_size[0] - img_x], dim=-1)  # (B, N_v, N_pc, 2) 2: h, w

        if normalize:
            # 归一化到 [-1， 1]
            img_points[..., 0] = (img_points[..., 0] / (render_size[0] - 1)) * 2 - 1
            img_points[..., 1] = (img_points[..., 1] / (render_size[1] - 1)) * 2 - 1

        return img_points

    def sample_point_features_from_2d(self, img_points, feature_2d):
        """
        使用 grid_sample 根据图像坐标提取点的 2D 特征，并进行视角平均。
        
        Args:
            img_points: (B, N_v, N_pc, 2), 图像坐标范围 [-1, 1]
            feature_2d: (B, N_v, C, H, W), 特征图

        Returns:
            (B, N_pc, C), 每个点的视角平均特征
        """
        B, N_v, N_pc, _ = img_points.shape
        _, _, C, H, W = feature_2d.shape

        # 1. 将 2D 坐标从像素坐标转换为 grid_sample 所需的归一化坐标 [-1, 1]
        # 假设 img_points 原本是像素坐标，需要归一化：
        # img_points[..., 0] in [0, W-1] -> [-1, 1]
        # img_points[..., 1] in [0, H-1] -> [-1, 1]
        # 否则如果已经归一化就跳过这步
        # img_points[..., 0] = (img_points[..., 0] / (W - 1)) * 2 - 1
        # img_points[..., 1] = (img_points[..., 1] / (H - 1)) * 2 - 1

        # 2. 构建 grid: (B*N_v, N_pc, 1, 2)
        grid = img_points.view(B * N_v, N_pc, 1, 2)  # (B*N_v, N_pc, 1, 2)

        # 3. 准备 features: (B*N_v, C, H, W)
        features = feature_2d.view(B * N_v, C, H, W)

        # 4. 采样: (B*N_v, C, N_pc, 1)
        sampled = F.grid_sample(features, grid, mode='bilinear', align_corners=True)  # (B*N_v, C, N_pc, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B*N_v, N_pc, C)

        # 5. reshape 回 (B, N_v, N_pc, C)
        sampled = sampled.view(B, N_v, N_pc, C)

        # 6. 对视角维度 N_v 平均 → (B, N_pc, C)
        point_features = sampled.mean(dim=1)

        return point_features

    def get_pixel_labels_from_gt(self, pixel_coords, masks, default_label=0):
        """
        Args:
            pixel_coords: (B, N_v, N_pc, 2), 取值范围在 [0, H-1], [0, W-1]
            masks: (B, N_v, 3, H, W)  # mask: mask rgb
            color2label: dict of {(r,g,b): label_idx}
            default_label: 默认标签
        Returns:
            labels: (B, N_pc, 1)
        """
        B, N_v, N_pc, _ = pixel_coords.shape
        _, _, _, H, W = masks.shape

        device = pixel_coords.device
        pixel_coords = pixel_coords.long()
        pixel_coords[..., 0] = pixel_coords[..., 0].clamp(0, H - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1].clamp(0, W - 1)

        # 准备展开索引
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_v, N_pc)
        view_idx = torch.arange(N_v, device=device).view(1, N_v, 1).expand(B, N_v, N_pc)
        h_idx = pixel_coords[..., 0]
        w_idx = pixel_coords[..., 1]

        # 获取对应像素的 RGB 值，输出形状：(B, N_v, N_pc, 3)
        rgb_vals = masks.permute(0,1,3,4,2)[batch_idx, view_idx, h_idx, w_idx]  # (B, N_v, N_pc, 3)

        # 初始化 label tensor
        label_tensor = torch.full((B, N_v, N_pc), default_label, dtype=torch.long, device=device)

        # 将 color2label 转换为 tensor 进行批量匹配
        color_list = torch.tensor(list(color2label.keys()), dtype=torch.uint8, device=device)  # (N_cls, 3)
        label_list = torch.tensor(list(color2label.values()), dtype=torch.long, device=device)  # (N_cls,)

        # 计算匹配掩码：逐颜色匹配 (B, N_v, N_pc, N_cls)
        rgb_vals_expand = rgb_vals.unsqueeze(-2)  # (B, N_v, N_pc, 1, 3)
        color_list_expand = color_list.view(1, 1, 1, -1, 3)  # (1, 1, 1, N_cls, 3)
        matches = (rgb_vals_expand == color_list_expand).all(dim=-1)  # (B, N_v, N_pc, N_cls)

        # 将匹配掩码转换为标签值（只保留第一个匹配）
        matched_label = matches @ label_list  # (B, N_v, N_pc)

        # 将匹配到的 label 替换到 label_tensor 中（没有匹配的保持 default_label）
        has_match = matches.any(dim=-1)  # (B, N_v, N_pc)
        label_tensor[has_match] = matched_label[has_match]

        # 对 N_v 维度进行众数投票
        mode_labels, _ = torch.mode(label_tensor, dim=1)  # (B, N_pc)

        return mode_labels.unsqueeze(-1)  # (B, N_pc, 1)

    def get_pixel_labels(self, pixel_coords, masks, default_label=0):
        """
        Args:
            pixel_coords: (B, N_v, N_pc, 2), 取值范围在 [0, H-1], [0, W-1]
            masks: (B, N_v, 17+1, H, W)  # mask: prob for each class
            default_label: 默认标签为牙龈
        Returns:
            labels: (B, N_pc, 1)
        """
        B, N_v, N_pc, _ = pixel_coords.shape
        masks = masks.view(B, N_v, -1, masks.shape[-2], masks.shape[-1])  # (B, N_v, C, H, W)
        _, _, _, H, W = masks.shape

        device = pixel_coords.device
        pixel_coords = pixel_coords.long()
        pixel_coords[..., 0] = pixel_coords[..., 0].clamp(0, H - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1].clamp(0, W - 1)

        # 准备展开索引
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_v, N_pc)
        view_idx = torch.arange(N_v, device=device).view(1, N_v, 1).expand(B, N_v, N_pc)
        h_idx = pixel_coords[..., 0]
        w_idx = pixel_coords[..., 1]

        # 获取每个点的类别 logits 或 prob: (B, N_v, N_pc, C)
        logits = masks.permute(0, 1, 3, 4, 2)[batch_idx, view_idx, h_idx, w_idx]  # (B, N_v, N_pc, C)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=1)  # (B, N_pc, C)
        
        with torch.no_grad():
            labels_int = avg_probs.argmax(dim=-1)  # (B, N_pc)
            labels_int = labels_int.masked_fill(labels_int == 17, default_label)

        # 用 one-hot 形式乘以 class index 实现 “离散前向 + 连续反向”
        one_hot = torch.nn.functional.one_hot(labels_int, num_classes=avg_probs.shape[-1]).float()
        label_indices = torch.arange(avg_probs.shape[-1], device=avg_probs.device).view(1, 1, -1)  # (1, 1, C)
        labels_final = (one_hot * label_indices).sum(dim=-1) + (avg_probs * label_indices).sum(dim=-1) - ((avg_probs * label_indices).sum(dim=-1)).detach()

        return labels_final.unsqueeze(-1)  # (B, N_pc, 1)
    

    def seg_model_2d_forward(self, inputs, batch_size=64):
        """
        Run a model on inputs in small batches to avoid OOM.
        Args:
            model: nn.Module
            inputs: Tensor of shape (N, C, H, W)
            batch_size: int
        Returns:
            outputs: list of model outputs for each batch, concatenated
        """
        outputs1, outputs2, outputs3 = [], [], []
        N = inputs.shape[0]
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = inputs[start:end]  # (B', C, H, W)
            out1, out2, out3 = self.seg_model_2d(batch)
            outputs1.append(out1)
            if out2 is not None:
                outputs2.append(out2)
            else:
                outputs2 = None
            if out3 is not None:
                outputs3.append(out3)
            else:
                outputs3 = None
        return (
            torch.cat(outputs1, dim=0),
            torch.cat(outputs2, dim=0) if outputs2 is not None else None,
            torch.cat(outputs3, dim=0) if outputs3 is not None else None
        )

    def forward(self, pc_coords, pc_coords_ori, pc_feats, renders=None, cameras_Rt=None, cameras_K=None):

        device = pc_feats.device
        bs = pc_coords.shape[0]
        n = pc_coords.shape[1]

        renders = rearrange(renders, 'b nv c h w -> (b nv) c h w') # (B, N_v, 3, H, W) -> (B*N_v, 3, H, W)
        render_size = renders.shape[-2:]

        # get 2d feature and 2d mask prediction
        if self.training:
            predict_2d_masks, predict_2d_aux, feature_2d = self.seg_model_2d(renders) # predict_2d_masks/predict_2d_aux: (B*N_v, 17+1, H, W), feature_2d: (B*N_v, C, H, W)
        else:
            predict_2d_masks, predict_2d_aux, feature_2d = self.seg_model_2d_forward(renders)
        # cameras_Rt (B, N_v, 4, 4), cameras_K (B, N_v, 3, 3)
        # 注意投影的时候点云坐标不能是标准化后的
        projected_pc = self.project_points(cameras_Rt, cameras_K, pc_coords_ori, render_size)  # (B, N_v, N_pc, 2)
        # point_features_2d = self.sample_point_features_from_2d(projected_pc, feature_2d) # (B, N_pc, C)
        point_features_2d = self.get_pixel_labels(projected_pc, predict_2d_masks) # (B, N_pc, 1)

        # 只取标准化后的点云坐标和法向量
        input_dict = {
            "coord": pc_coords,  # (B, N_pc, 3)
            "feat": pc_feats,    # (B, N_pc, 3)
            "grid_size": self.grid_size,  # 网格大小
            "point_to_pixel_feat": point_features_2d
        }
        predict_pc_labels, predict_pc_boundary_labels, pc_feat = self.seg_model_3d(input_dict) # predict_pc_labels: (B, 17, N_pc)

        pc_coords = torch.cat([pc_coords[i] for i in range(bs)], dim=0).contiguous()
        pc_offsets = torch.tensor([n * (i + 1) for i in range(bs)], device=device, dtype=torch.int32)
        cbl_loss_aux = [pc_coords, pc_feat, pc_offsets]
        
        return predict_pc_labels, predict_pc_boundary_labels, predict_2d_masks, predict_2d_aux, cbl_loss_aux
    

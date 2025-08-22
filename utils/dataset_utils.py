import torch

def custom_collate_fn(batch):
        coords = []
        coords_ori = []
        feats = []
        labels = []
        boundary_labels = []
        vertices_list = []
        faces = []
        renders = []
        masks = []
        cameras_Rt = []
        cameras_K = []
        file_names = []

        for data_dict in batch:
            # 解包数据字典
            coord = data_dict['coord']
            coord_ori = data_dict['coord_ori']
            feat = data_dict['feat']
            label = data_dict['label']
            boundary_label = data_dict['boundary_label']
            p_coords = data_dict['vertice']
            face = data_dict['face']
            render = data_dict['render']
            mask = data_dict['mask']
            camera_Rt = data_dict['camera_Rt']
            camera_K = data_dict['camera_K']
            file_name = data_dict['file_name']

            coords.append(coord)
            coords_ori.append(coord_ori)
            feats.append(feat)
            labels.append(label)
            boundary_labels.append(boundary_label)
            vertices_list.append(p_coords)  # 不堆叠，保留为 list of np.array
            faces.append(face) # 保留为 list of np.array
            renders.append(render)
            masks.append(mask)
            cameras_Rt.append(camera_Rt)
            cameras_K.append(camera_K)
            file_names.append(file_name) # 保留为 list of str

        # 堆叠固定 shape 的数据
        feats = torch.stack(feats)  # (B, num_points, 6)
        coords = torch.stack(coords)  # (B, num_points, 3)
        coords_ori = torch.stack(coords_ori)  # (B, num_points, 3)
        labels = torch.stack(labels)            # (B, num_points)
        boundary_labels = torch.stack(boundary_labels)  # (B, num_points)
        renders = torch.stack(renders)          # (B, num_views, 3, H, W)
        masks = torch.stack(masks)              # (B, num_views, H, W)
        cameras_Rt = torch.stack(cameras_Rt)    # (B, num_views, 4, 4)
        cameras_K = torch.stack(cameras_K)      # (B, num_views, 3, 3)

        return_dict = {
            'pc_feats': feats,
            'pc_coords': coords,
            'pc_coords_ori': coords_ori,
            'labels': labels,
            'boundary_labels': boundary_labels,
            'vertices': vertices_list,  # 保持为 list of np.array
            'faces': faces,
            'renders': renders,
            'masks': masks,
            'cameras_Rt': cameras_Rt,
            'cameras_K': cameras_K,
            'file_names': file_names
        }
        return return_dict
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from einops import rearrange

from loss.cbl import CBLLoss
from models.main_network import ToothSegNet
from dataset.teeth3ds import Teeth3DSDataset
from utils.other_utils import output_pred_ply, save_metrics_to_txt
from utils.color_utils import label2color_lower, label2color_upper
from utils.metric_utils import calculate_miou, calculate_biou
from utils.dataset_utils import custom_collate_fn


class ToothSegmentationPipeline:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        self.best_miou = 0.0
        self.best_epoch = 0

        self.get_dataloader()


        self.build_model(num_classes=17, use_pretrain=self.args.load_ckp)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-6)
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_cbl = CBLLoss().to(self.device)

        if args.use_wandb:
            self.logger = wandb.init(entity='3dtooth_seg',project='3dtooth_seg', name=args.save_dir, dir=args.save_dir, config=vars(args))
        else:
            self.logger = None

    def get_dataloader(self):
        # upper_files = glob.glob(os.path.join(self.args.data_dir, 'upper', '*.ply'))
        # lower_files = glob.glob(os.path.join(self.args.data_dir, 'lower', '*.ply'))
        # file_list = upper_files + lower_files
        # print(f"Found {len(file_list)} ply files in {self.args.data_dir}")

        train_dataset = Teeth3DSDataset(
            root=self.args.data_dir, in_memory=False,
            force_process=False, train_test_split=self.args.train_test_split, is_train=True,
            num_points=self.args.num_points, sample_points=self.args.sample_points, sample_views=self.args.sample_views
        )
        test_dataset = Teeth3DSDataset(
            root=self.args.data_dir, in_memory=False,
            force_process=False, train_test_split=self.args.train_test_split, is_train=False,
            num_points=self.args.num_points, sample_points=self.args.sample_points, sample_views=self.args.sample_views
        )

        print(f"Dataset size: Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
        )

        self.test_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
        )


    def build_model(self, num_classes, use_pretrain):
        self.model = ToothSegNet(
            in_channels=6, num_classes=num_classes, use_pretrain=use_pretrain).to(self.device)

    def train(self):
        
        self.model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            loop = tqdm(self.train_dataloader, desc=f"Epoch [{epoch+1}/{self.args.epochs}]", leave=False)

            for batch_idx, batch_data in enumerate(loop):
                pc_feats = batch_data['pc_feats'].to(self.device)
                pc_coords = batch_data['pc_coords'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                boundary_labels = batch_data['boundary_labels'].to(self.device)
                renders = batch_data['renders'].to(self.device)
                masks = batch_data['masks'].to(self.device)
                cameras_Rt = batch_data['cameras_Rt'].to(self.device)
                cameras_K = batch_data['cameras_K'].to(self.device)

                self.optimizer.zero_grad()

                predict_pc_labels, predict_pc_boundary_labels, predict_2d_masks, predict_2d_aux, cbl_loss_aux = self.model(pc_coords, pc_feats, renders, cameras_Rt, cameras_K)
                
                # calculate losses
                loss_3d = self.criterion_ce(predict_pc_labels, labels)
                loss_3d_boundary = self.criterion_ce(predict_pc_boundary_labels, boundary_labels)

                masks = rearrange(masks, 'b nv h w -> (b nv) h w') # (B, N_v, H, W) -> (B*N_v, H, W)
                loss_2d = self.criterion_ce(predict_2d_masks, masks) + self.criterion_ce(predict_2d_aux, masks)

                loss_cbl = self.criterion_cbl(cbl_loss_aux, labels.view(-1))
                loss = loss_2d + loss_3d + loss_3d_boundary + loss_cbl


                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                loop.set_postfix(loss=loss.item())

                if self.logger:
                    self.logger.log({
                        "batch_loss": loss.item(),
                        "loss_3d": loss_3d.item(),
                        "loss_3d_boundary": loss_3d_boundary.item(),
                        "loss_2d": loss_2d.item(),
                        "loss_cbl": loss_cbl.item(),
                        "step": epoch * len(self.train_dataloader) + batch_idx
                    })

            avg_loss = total_loss / len(self.train_dataloader)
            tqdm.write(f"Epoch [{epoch+1}/{self.args.epochs}] - Average Loss: {avg_loss:.4f}")
            self.scheduler.step()

            if self.logger:
                self.logger.log({"epoch": epoch + 1, "epoch_loss": avg_loss, "lr": self.optimizer.param_groups[0]['lr']})


            if (epoch + 1) % self.args.eval_epoch_step == 0 or (epoch + 1) == self.args.epochs:

                self.predict(epoch)
                self.model.train()  # Reset to training mode after evaluation

    def predict(self, current_epoch=None, save_result=False):


        self.model.eval()

        lower_palette = np.array(
        [[label2color_lower[label][2][0],
          label2color_lower[label][2][1],
          label2color_lower[label][2][2]]
         for label in range(0, 17)], dtype=np.uint8) # (17, 3)
        
        upper_palette = np.array(
        [[label2color_upper[label][2][0],
          label2color_upper[label][2][1],
          label2color_upper[label][2][2]]
         for label in range(0, 17)], dtype=np.uint8)
        
        miou = []
        biou = []
        per_class_iou = []
        merge_iou = []


        with torch.no_grad():
            loop_val = tqdm(self.test_dataloader, desc=f"Validating", leave=False)
            for batch_idx, batch_data in enumerate(loop_val):
                pc_feats = batch_data['pc_feats'].to(self.device)
                pc_coords = batch_data['pc_coords'].to(self.device)
                point_coords = batch_data['vertices']
                face_info = batch_data['faces']
                file_name = batch_data['file_names']
                labels = batch_data['labels'].to(self.device)
                boundary_labels = batch_data['boundary_labels'].to(self.device)

                renders = batch_data['renders'].to(self.device)
                cameras_Rt = batch_data['cameras_Rt'].to(self.device)
                cameras_K = batch_data['cameras_K'].to(self.device)


                point_seg_result, point_seg_boundary_result, _, _, _ = self.model(pc_coords, pc_feats, renders, cameras_Rt, cameras_K) # point_seg_result: (B, num_classes, N_pc)
                # pred_softmax = torch.nn.functional.softmax(point_seg_result, dim=1)
                # _, pred_classes = torch.max(pred_softmax, dim=1) # (B, N_pc)

                # calculate miou
                pred_classes = point_seg_result.argmax(dim=1)  # (B, N_pc)
                pred_classes[pred_classes == 17] = 0
                pred_classes[pred_classes == 18] = 0

                pred_boundary = point_seg_boundary_result.argmax(dim=1)
                
                miou_batch, per_class_iou_batch, merge_iou_batch = calculate_miou(
                    pred_classes, labels, n_class=17)
                biou_batch = calculate_biou(pred_boundary, boundary_labels, n_class=2)

                miou.append(miou_batch)
                biou.append(biou_batch)
                per_class_iou.append(per_class_iou_batch)
                merge_iou.append(merge_iou_batch)

                if save_result:
                    pred_classes = pred_classes.cpu().numpy().astype(np.uint8)

                    bs = len(point_coords)
                    for i in range(bs):
                        file_name_i = file_name[i]
                        if 'upper' in file_name_i:
                            save_path = os.path.join(self.args.save_predict_mask_dir, 'upper', f"{file_name_i}_predict.ply")
                            pred_mask = upper_palette[pred_classes[i]]
                        elif 'lower' in file_name_i:
                            save_path = os.path.join(self.args.save_predict_mask_dir, 'lower', f"{file_name_i}_predict.ply")
                            pred_mask = lower_palette[pred_classes[i]]
                        output_pred_ply(pred_mask, None, save_path, point_coords[i], face_info[i])


                    print(f"Predict end, result saved at {self.args.save_predict_mask_dir}")
                
                loop_val.set_postfix(miou=miou_batch)
            

            miou = torch.stack(miou).mean().item()
            biou = torch.stack(biou).mean().item()

            per_class_iou = torch.stack(per_class_iou).mean(dim=0) # (C, )
            merge_iou = torch.stack(merge_iou).mean(dim=0) # (num_pairs, )


            if self.logger:
                self.logger.log({"val_miou": miou})

            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = current_epoch + 1 if current_epoch is not None else 'N/A'
                print(f"New best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch}")
            
            if current_epoch is not None:
                if miou > self.best_miou or (current_epoch + 1) == self.args.epochs:
                    # save model
                    save_path = os.path.join(self.args.save_dir, f"toothseg_epoch{current_epoch+1}_miou{miou:.3f}.pth")
                    torch.save({
                        'epoch': current_epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)
                    print(f"Saved model checkpoint to {save_path}")

                    # save metrics
                    save_metrics_to_txt(
                        filepath=os.path.join(self.args.save_dir, f"metrics_result_epoch{current_epoch+1}_miou{miou:.3f}.txt"),
                        num_classes=17,
                        miou=miou,
                        biou=biou,
                        per_class_miou=per_class_iou.cpu().numpy(),
                        merge_iou=merge_iou.cpu().numpy(),
                        )


    # def custom_collate_fn(self, batch):
    #     pointclouds = []
    #     labels = []
    #     point_coords_list = []
    #     face_infos = []
    #     renders = []
    #     masks = []
    #     cameras_Rt = []
    #     cameras_K = []
    #     file_names = []

    #     for data_dict in batch:
    #         # 解包数据字典
    #         pc = data_dict['pointcloud']
    #         label = data_dict['labels']
    #         p_coords = data_dict['point_coords']
    #         f_info = data_dict['face_info']
    #         render = data_dict['renders']
    #         mask = data_dict['masks']
    #         camera_Rt = data_dict['cameras_Rt']
    #         camera_K = data_dict['cameras_K']
    #         file_name = data_dict['file_names']

    #         pointclouds.append(pc)
    #         labels.append(label)
    #         point_coords_list.append(p_coords)  # 不堆叠，保留为 list of np.array
    #         face_infos.append(f_info) # 保留为 list of np.array
    #         renders.append(render)
    #         masks.append(mask)
    #         cameras_Rt.append(camera_Rt)
    #         cameras_K.append(camera_K)
    #         file_names.append(file_name) # 保留为 list of str

    #     # 堆叠固定 shape 的数据
    #     pointclouds = torch.stack(pointclouds)  # (B, num_points, 6)
    #     labels = torch.stack(labels)            # (B, num_points)
    #     renders = torch.stack(renders)          # (B, num_views, 3, H, W)
    #     masks = torch.stack(masks)              # (B, num_views, H, W)
    #     cameras_Rt = torch.stack(cameras_Rt)    # (B, num_views, 4, 4)
    #     cameras_K = torch.stack(cameras_K)      # (B, num_views, 3, 3)

    #     return_dict = {
    #         'pointclouds': pointclouds,
    #         'labels': labels,
    #         'point_coords': point_coords_list,  # 保持为 list of np.array
    #         'face_infos': face_infos,
    #         'renders': renders,
    #         'masks': masks,
    #         'cameras_Rt': cameras_Rt,
    #         'cameras_K': cameras_K,
    #         'file_names': file_names
    #     }
    #     return return_dict
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.datasets/teeth3ds')
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--sample_views', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='exp/train')
    parser.add_argument('--eval_epoch_step', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_test_split', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--load_ckp', type=str, default=None, help='Use trained checkpoint path')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_predict_mask_dir = os.path.join(args.save_dir, 'predict_masks')
    os.makedirs(os.path.join(args.save_predict_mask_dir, 'upper'), exist_ok=True)
    os.makedirs(os.path.join(args.save_predict_mask_dir, 'lower'), exist_ok=True)
    
    

    pipeline = ToothSegmentationPipeline(args)
    
    if args.train:
        print("Starting training...")
        pipeline.train()
    else:
        print("Starting prediction...")
        pipeline.predict(save_result=False)
    
    
    
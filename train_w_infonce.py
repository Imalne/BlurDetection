from model_arch import BlurDetector
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data import BlurDataset, train_aug_paired, train_aug_input, val_aug
from loss import PixelMultiClassInfoNCELoss
import torch.nn.functional as F

import os

from torch.utils.tensorboard import SummaryWriter

train_set = BlurDataset(
    "/mnt/c/Users/WhIma/Desktop/CUHK/image/", 
    "/mnt/c/Users/WhIma/Desktop/CUHK/gt", 
    aug=train_aug_paired,
    aug_for_input=train_aug_input
    )
val_set   = BlurDataset(
    "/mnt/c/Users/WhIma/Desktop/test/image/", 
    "/mnt/c/Users/WhIma/Desktop/test/gt", 
    aug=val_aug
    )

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=20, shuffle=False, num_workers=4, pin_memory=True)

val_interval = len(train_loader) * 5
save_dir = "./logs_infonce/"

model = BlurDetector(out_classes=3).cuda()
opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
loss_infonce = PixelMultiClassInfoNCELoss(temperature=0.1, max_samples=4096)

iter_count = -1

os.makedirs(save_dir, exist_ok=True)

tb_logger = SummaryWriter(log_dir=save_dir)

for epoch in range(350):
    if epoch == 200:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    for img, gt in train_loader:  # gt: (H,W) blur=1, clear=0
        img, gt = img.cuda(), gt.cuda()
        out_up, out_ds, proj = model(img, proj=True)
        gt_ds = F.interpolate(gt.unsqueeze(1).float(), size=out_ds.shape[2:], mode='nearest').long().squeeze(1)
        Ld = loss_fn(out_ds, gt_ds)
        Lu = loss_fn(out_up, gt)
        loss = Lu + 4 * Ld
        # InfoNCE loss
        Linfo = loss_infonce(proj, gt_ds)

        loss = loss + Linfo * 0.1

        opt.zero_grad()
        loss.backward()
        opt.step()
        iter_count += 1

        print(f"Epoch [{epoch}/{350}] Iter [{iter_count}] Loss: {loss.item():.4f}; Ld: {Ld.item():.4f}; Lu: {Lu.item():.4f}, Linfo: {Linfo.item():.4f}")
        tb_logger.add_scalar("Train/Ld", Ld.item(), iter_count)
        tb_logger.add_scalar("Train/Lu", Lu.item(), iter_count)
        tb_logger.add_scalar("Train/Linfo", Linfo.item(), iter_count)
        tb_logger.add_scalar("Train/Loss", loss.item(), iter_count)

        if iter_count % val_interval == 0:
            model.eval()
            total_correct = 0
            total_pixels  = 0

            with torch.no_grad():
                for vimg, vgt in val_loader:
                    with torch.no_grad():
                        vimg, vgt = vimg.cuda(), vgt.cuda()
                        vout, _ = model(vimg)
                        preds = torch.argmax(vout, dim=1)

                        total_correct += (preds == vgt).sum().item()
                        total_pixels  += torch.numel(vgt)

            acc = total_correct / total_pixels
            mae = torch.abs(preds.float() - vgt.float()).mean().item()
            precision = (torch.logical_and(preds==1, vgt==1).sum().item()) / ( (preds==1).sum().item() + 1e-8)
            recall = (torch.logical_and(preds==1, vgt==1).sum().item()) / ( (vgt==1).sum().item() + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            print(f"Validation Accuracy after {iter_count} iterations: {acc*100:.2f}%")
            print(f"Validation MAE after {iter_count} iterations: {mae:.4f}")
            print(f"Validation Precision after {iter_count} iterations: {precision*100:.2f}%")
            print(f"Validation Recall after {iter_count} iterations: {recall*100:.2f}%")
            print(f"Validation F1-Score after {iter_count} iterations: {f1_score*100:.2f}%")
            tb_logger.add_scalar("Val/MAE", mae, iter_count)
            tb_logger.add_scalar("Val/Precision", precision, iter_count)
            tb_logger.add_scalar("Val/Recall", recall, iter_count)
            tb_logger.add_scalar("Val/F1_Score", f1_score, iter_count)
            tb_logger.add_scalar("Val/Accuracy", acc, iter_count)            
            torch.save(model.state_dict(), f"{save_dir}/model_iter_{iter_count}.pth")
            

            visualize = True
            if visualize:
                import cv2
                import numpy as np
                vimg_np = (vimg.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)
                vimg_np = np.concatenate([vimg_np[i] for i in range(vimg_np.shape[0])], axis=1)
                # vgt to one-hot
                vgt_one_hot = (F.one_hot(vgt, num_classes=3) * 255).cpu().numpy().astype(np.uint8)
                vgt_one_hot = np.concatenate([vgt_one_hot[i] for i in range(vgt_one_hot.shape[0])], axis=1)
                # print(vgt_one_hot.shape)
                vout_soft = (F.softmax(vout, dim=1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                vout_soft = np.concatenate([vout_soft[i] for i in range(vout_soft.shape[0])], axis=1)


                # print(vout_soft.shape)
                # vgt_np  = (vgt[0].cpu().numpy()*127).astype(np.uint8)
                # print(vgt_np.shape)
                # pred_np = (preds[0].cpu().numpy()*127).astype(np.uint8)
                # print(pred_np.shape)
                visualize_img = np.concatenate([vimg_np, vgt_one_hot, vout_soft], axis=0)  # to BGR for cv2
                cv2.imwrite(f"{save_dir}/val_iter_{iter_count}.png", visualize_img)

            model.train()

# StereoSpike QAT + Distillation Training Script (SQAKD-style)

import os
import time
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from spikingjelly.clock_driven import functional, surrogate
from torchvision import transforms

from network.metrics import MeanDepthError, OnePixelAccuracy, depth_to_disparity

from network.SNN_models_simpquant import (
    SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3,
    SQAKD_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3,
    SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3
)

##############################
# Packages for SQAKD
# Parameters
# Original setting: KD_GAMMA: 0, KD_ALPHA: 1, KD_BETA:0 
# KD_GAMMA for loss for stereospike itself
# KD_ALPHA for loss for kownledge dis
##############################
from network.custom_modules import QConv
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomTimeMirror
KD_T = 4
KD_GAMMA = 0
KD_ALPHA = 1
KD_BETA = 0.0

# Sterospike
from network.loss import Total_Loss

##############################
# Global Parameters
##############################
NUM_EPOCHS = 70
SEED = 4064822634


##############################
# Loss for KL
##############################
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

##############################
# PLOT #
##############################
def plot_training_progress(train_loss_list, train_mde_list, val_mde_list):
    import matplotlib.pyplot as plt
    epochs = list(range(len(train_loss_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, train_mde_list, label='Train MDE')
    plt.plot(epochs, val_mde_list, label='Val MDE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/training_progress_{timestamp}.pdf")
    plt.close()


##############################
# TRAIN #
##############################
def train(train_loader, val_loader, criterions,  n_epochs=70, seed = 4064822634):
    best_mde = float('inf')
    total_iter = 0
    train_loss_list = []
    train_mde_list = []
    val_mde_list = []
    criterion_cls = criterions[0]
    criterion_div = criterions[1]
    criterion_kd = criterions[2] 
    for epoch in range(n_epochs):
        model_s.train()
        model_t.eval()

        running_loss, running_MDE, running_OPA = 0., 0., 0.

        for i, (init_pots, wL, wR, tL, tR, label) in enumerate(train_loader):
            wL, wR, tL, tR, label = wL.to(device, dtype=torch.float), wR.to(device, dtype=torch.float), tL.to(device, dtype=torch.float), tR.to(device, dtype=torch.float), label.to(device)
            _, train_chunks = model_s.reformat_input_data(wL, wR, tL, tR)
            binary_chunks = (train_chunks >= 1).to(train_chunks.dtype)

            functional.reset_net(model_s)
            functional.reset_net(model_t)

            optimizer_m.zero_grad()
            optimizer_q.zero_grad()

            pred_s, spks_s = model_s(binary_chunks)
            with torch.no_grad():
                pred_t, _ = model_t(binary_chunks)

            logit_s = pred_s[0].view(pred_s[0].size(0), -1)
            logit_t = pred_t[0].view(pred_t[0].size(0), -1)

            loss_cls = criterion_cls(pred_s, label, spks_s)
            loss_div = criterion_div(logit_s, logit_t)
            loss_kd = 0 # Don't need for kd

            loss_total = KD_GAMMA * loss_cls + KD_ALPHA * loss_div + KD_BETA * loss_kd

            print("i:{} loss:{}".format(i, loss_total.item()))

            if i == 0:
                print(f"gamma: {KD_GAMMA}, alpha: {KD_ALPHA}, kd_beta: {KD_BETA}")

            loss_total.backward()
            optimizer_m.step()
            optimizer_q.step()

            writer.add_scalar('train/loss_cls', loss_cls.item(), total_iter)
            writer.add_scalar('train/loss_div', loss_div.item(), total_iter)
            writer.add_scalar('train/loss_total', loss_total.item(), total_iter)

            lin_pred = pred_s[0]
            lin_label = label

            MDE = MeanDepthError(lin_pred, lin_label)
            OPA = OnePixelAccuracy(depth_to_disparity(lin_pred), depth_to_disparity(lin_label))

            running_loss += loss_total.item()
            running_MDE += MDE
            running_OPA += OPA
            total_iter += 1

        scheduler_m.step()
        scheduler_q.step()

        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('MDE/train', running_MDE, epoch)
        writer.add_scalar('OPA/train', running_OPA, epoch)

        train_loss_list.append(running_loss)
        train_mde_list.append(running_MDE)

        print(f"[Epoch {epoch}] Loss={running_loss:.3f}, MDE={running_MDE:.3f}, 1PA={running_OPA:.3f}")

        # Validation evaluation
        model_s.eval()
        total_mde, total_opa = 0., 0.
        with torch.no_grad():
            for (init_pots, wL, wR, tL, tR, label) in val_loader:
                wL, wR, tL, tR, label = wL.to(device, dtype=torch.float), wR.to(device, dtype=torch.float), tL.to(device, dtype=torch.float), tR.to(device, dtype=torch.float), label.to(device)
                _, val_chunks = model_s.reformat_input_data(wL, wR, tL, tR)
                binary_chunks = (val_chunks >= 1).to(val_chunks.dtype)

                pred, _ = model_s(binary_chunks)
                lin_pred = pred[0]
                lin_label = label

                mde = MeanDepthError(lin_pred, lin_label)
                opa = OnePixelAccuracy(depth_to_disparity(lin_pred), depth_to_disparity(lin_label))

                total_mde += mde
                total_opa += opa

        avg_mde = total_mde / len(val_loader)
        avg_opa = total_opa / len(val_loader)
        writer.add_scalar('MDE/val', avg_mde, epoch)
        writer.add_scalar('OPA/val', avg_opa, epoch)
        val_mde_list.append(avg_mde)

        print(f"[Validation] MDE={avg_mde:.3f}, 1PA={avg_opa:.3f}")

        if avg_mde < best_mde:
            best_mde = avg_mde
            torch.save(model_s.state_dict(), "./results/checkpoints/SQAKD_binaryinp_simplified_stereospike_seed{}.pth".format(seed))
            print(f"\tSaved best model at epoch {epoch} with MDE {avg_mde:.4f}")
    
    plot_training_progress(train_loss_list, train_mde_list, val_mde_list)

##############################
# TEST #
##############################
def test(test_loader):
    model_s.eval()
    total_mde, total_opa = 0., 0.
    with torch.no_grad():
        for (init_pots, wL, wR, tL, tR, label) in test_loader:
            wL, wR, tL, tR, label = wL.to(device, dtype=torch.float), wR.to(device, dtype=torch.float), tL.to(device, dtype=torch.float), tR.to(device, dtype=torch.float), label.to(device)
            _, test_chunks = model_s.reformat_input_data(wL, wR, tL, tR)
            binary_chunks = (test_chunks >= 1).to(test_chunks.dtype)

            pred, _ = model_s(binary_chunks)
            lin_pred = pred[0]
            lin_label = label

            mde = MeanDepthError(lin_pred, lin_label)
            opa = OnePixelAccuracy(depth_to_disparity(lin_pred), depth_to_disparity(lin_label))

            total_mde += mde
            total_opa += opa

    avg_mde = total_mde / len(test_loader)
    avg_opa = total_opa / len(test_loader)
    print(f"[Test] MDE={avg_mde:.3f}, 1PA={avg_opa:.3f}")
    


if __name__ == '__main__':

    # --- Loss setup --- #
    penal = False
    penal_beta = 0.

    # --- Device setup --- #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load datasets --- #
    tsfs = transforms.Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomTimeMirror(p=0.5)
    ])

    train_set = torch.load('./datasets/train_set.pt')
    val_set = torch.load('./datasets/val_set.pt')
    test_set = torch.load('./datasets/test_set.pt')

    batchsize = 1  # or override later via args
    train_data_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, drop_last=True, pin_memory=True)
    val_data_loader   = DataLoader(dataset=val_set, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)
    test_data_loader  = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # --- Initialize models --- #
    model_s = SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3(
        input_chans=4, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True,
        multiply_factor=10., surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)

    model_t = SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3(
        input_chans=4, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True,
        multiply_factor=10., surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)

    model_t.load_state_dict(torch.load("./results/checkpoints/stereospike_seed{}.pth".format(SEED)))
    model_t.eval()

    # --- Param separation --- #
    model_params = []
    quant_params = []
    trainable_params = list(model_s.parameters())
    for m in model_s.modules():
        if isinstance(m, QConv):
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            if m.quan_weight:
                quant_params.append(m.lW)
                quant_params.append(m.uW)
            if m.quan_act:
                quant_params.append(m.lA)
                quant_params.append(m.uA)
                quant_params.append(m.lA_t)
                quant_params.append(m.uA_t)
            if m.quan_act or m.quan_weight:
                quant_params.append(m.output_scale)
            print("QConv", m)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            model_params.append(m.weight)
            if m.bias is not None:
                model_params.append(m.bias)
            print("nn", m)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                model_params.append(m.weight)
                model_params.append(m.bias)

    print("# total params:", sum(p.numel() for p in trainable_params))
    print("# model params:", sum(p.numel() for p in model_params))
    print("# quantizer params:", sum(p.numel() for p in quant_params))

    optimizer_m = torch.optim.Adam(model_params, lr=2e-4)
    optimizer_q = torch.optim.Adam(quant_params, lr=1e-5)

    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=[8, 42, 60], gamma=0.5)
    scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=[8, 42, 60], gamma=0.5)

    # --- Losses using SQAKD's utils_distill --- #
    num_training_data = len(train_set)
    criterions = []
    criterion_cls = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=penal, beta=penal_beta)
    criterions.append(criterion_cls)
    criterion_div = DistillKL(KD_T)
    criterions.append(criterion_div)
    criterion_kd = DistillKL(KD_T)
    criterions.append(criterion_kd)
    
    writer = SummaryWriter("./runs/SQAKD_QAT")

    train(train_data_loader, val_data_loader, criterions, n_epochs=NUM_EPOCHS, seed=SEED)
    test(test_data_loader)




    
# In[ ]:
import sys
import yaml
from tqdm import tqdm
import warnings, re
warnings.filterwarnings("ignore", message=r"Trying to call `reset$begin:math:text$$end:math:text$`.*MemoryModule")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import surrogate, functional

from network.metrics import MeanDepthError, OnePixelAccuracy, depth_to_disparity, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss

from network.SNN_models_simpquant_v100 import (
    SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3,
    SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v4
)

from network.custom_modules import QConv, QConv_DW, QConv_PW

import math

from spikingjelly.activation_based import base

def safe_reset(net: torch.nn.Module):
    for m in net.modules():
        if isinstance(m, base.MemoryModule):
            m.reset()

# In[ ]:
device = (
    "cuda" if torch.cuda.is_available() # GPU
    else "mps" if torch.backends.mps.is_available() # Apple M Series
    else "cpu"
)

print(f'Running on {device}\n')

# In[ ]:
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

cfg = load_config('./configuration_SQAKD_v25.yaml')
print(cfg)

# In[ ]:
torch.manual_seed(cfg['training']['seed'])
print('Setting random seed to', cfg['training']['seed'])

# In[ ]:
# Remove the conflicting datasets module from sys.modules if it exists
# I had errors due to the huggingface library installed on my env,
# feel free to remove this snippet if you don't need this workaround.
if 'datasets' in sys.modules:
    del sys.modules['datasets']

train_set = torch.load(cfg['datasets']['training_set'], weights_only=False)
val_set = torch.load(cfg['datasets']['validation_set'], weights_only=False)

# In[ ]:
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    pin_memory=True
)

val_data_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    pin_memory=True
)

# In[ ]:
# Instanciate and load the pretrained weights of the teacher model

full_precision_model = SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3(
    input_chans=4,
    tau=cfg['neuron']['tau'],
    v_threshold=cfg['neuron']['v_threshold'],
    v_reset=cfg['neuron']['v_reset'],
    use_plif= cfg['neuron']['use_plif'],
    multiply_factor=10.,
    surrogate_function=surrogate.ATan(),
    learnable_biases=False
).to(device)

full_precision_model.load_state_dict(torch.load(cfg['model']['checkpoint'], map_location=device))

# In[ ]:
# Instanciate the student model

quantized_model = SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v4(
    input_chans=4,
    tau=cfg['neuron']['tau'],
    v_threshold=cfg['neuron']['v_threshold'],
    v_reset=cfg['neuron']['v_reset'],
    use_plif=cfg['neuron']['use_plif'],
    multiply_factor=100.0,
    surrogate_function=surrogate.ATan(),
    learnable_biases=False,
    multiply_ratio=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,10.0,10.0,10.0,10.0]
).to(device)


# In[ ]:
full_precision_model.eval()
for p in full_precision_model.parameters():
    p.requires_grad_(False)

# In[ ]:
model_params = []
quant_params = []
trainable_params = list(quantized_model.parameters())
for m in quantized_model.modules():
    if isinstance(m, QConv) or isinstance(m, QConv_DW) or isinstance(m, QConv_PW):
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
        # print("QConv", m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        model_params.append(m.weight)
        if m.bias is not None:
            model_params.append(m.bias)
        # print("nn", m)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.affine:
            model_params.append(m.weight)
            model_params.append(m.bias)

print("# total params:", sum(p.numel() for p in trainable_params))
print("# model params:", sum(p.numel() for p in model_params))
print("# quantizer params:", sum(p.numel() for p in quant_params))

# In[ ]:
# KL Loss function could be somewhat tricky with StereoSpike's framework
# because the model is not creating labels that we can apply softmax on but rather 
# depth estimation on each pixel.
# What I suggest is to divide the depth maps in K bins,
# to form some form of "labels" with K big enough to keep most of the information.
# And apply softmax on that to better match with the original paper.

""" #### Geoffroy's Version 
def depths_to_bin_probs(depth_map, bin_edges):
    
    #depth_map: [B, 1, H, W] continuous depths
    #bin_edges: [K+1] tensor of bin boundaries (including min & max)
    #Returns: [B, K, H, W] probability maps (1-hot hard bins here)
    
    # Digitize: find which bin each pixel falls into
    # Note: torch.bucketize returns bin indices in [0, K]
    bin_indices = torch.bucketize(depth_map, bin_edges) - 1  # shift to [0, K-1]
    bin_indices = bin_indices.clamp(0, len(bin_edges)-2)     # clamp just in case
    # One-hot encode
    B, _, H, W = depth_map.shape
    K = len(bin_edges) - 1
    one_hot = F.one_hot(bin_indices.squeeze(1), num_classes=K)  # [B, H, W, K]
    one_hot = one_hot.permute(0, 3, 1, 2).float()               # [B, K, H, W]

    return one_hot

# In[ ]:
class DepthBinKLLoss(torch.nn.Module):
    def __init__(self, bin_edges, temperature=1.0):
        super().__init__()
        self.register_buffer('bin_edges', bin_edges)
        self.T = temperature

    def forward(self, depth_s, depth_t):
        # Convert both to one-hot bin distributions
        prob_t = depths_to_bin_probs(depth_t.detach(), self.bin_edges)
        prob_s = depths_to_bin_probs(depth_s, self.bin_edges)

        # Apply temperature scaling
        log_prob_s = torch.log(prob_s + 1e-8)  # avoid log(0)
        loss = F.kl_div(log_prob_s, prob_t, reduction='batchmean') * (self.T ** 2)
        return loss
"""

# --- Remove or disable the existing one-hot function and KL loss ---
def gumbel_soft_bucketize(values, bin_edges, temperature=0.5):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dist = -(values[..., None] - bin_centers) ** 2
    # Apply Gumbel-Softmax
    return F.gumbel_softmax(dist, tau=temperature, hard=False, dim=-1)

class DepthBinKLLoss(torch.nn.Module):
    def __init__(self, bin_edges, temperature=1.0):
        super().__init__()
        self.register_buffer('bin_edges', bin_edges)
        self.T = temperature

    def forward(self, depth_s, depth_t):
        # Convert both to one-hot bin distributions
        prob_t = gumbel_soft_bucketize(depth_t.detach(), self.bin_edges, temperature=self.T) # larger temperature
        prob_s = gumbel_soft_bucketize(depth_s, self.bin_edges, temperature=self.T)

        # Apply temperature scaling
        log_prob_s = torch.log(prob_s.clamp(min=1e-6)) #torch.log(prob_s + 1e-8)  # avoid log(0)
        loss = F.kl_div(log_prob_s, prob_t, reduction='batchmean') * (self.T ** 2)
        return loss

# Bin edges might be modified, I'm not sure how many we want and the boundaries so I'll let you guys set how many you
# want but this should be ok.
bin_edges = torch.linspace(0.0, 10.0, steps=801)  # 80 bins ###### advice incrasing the bin_steps
criterion_kd = DepthBinKLLoss(bin_edges, temperature=cfg['sqakd']['temperature']).to(device) #### Geoffroy's Version

# In[ ]:
epochs = cfg['training']['epochs']
lr = cfg['training']['learning_rate']
weight_decay = cfg['training']['weight_decay']


# Optimizer for model weights
optimizer_m = torch.optim.AdamW(
    model_params,
    lr=lr,
    weight_decay=weight_decay
)

# Optimizer for the quantization process
optimizer_q = torch.optim.AdamW(
    quant_params,
    lr=lr,
) # Let's try with no weight decay for the quantization

# Scheduler
scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=epochs)
scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=epochs)

# In[ ]:
x = torch.rand((1,1,4,260,346)).to(device)
pred_s, _ = quantized_model(x)
print(pred_s[0].shape)


pred_t, _ = full_precision_model(x)
print(pred_t[0].shape)

criterion_kd(pred_s[0], pred_t[0])

# In[ ]:
# Tensorboard logger
writer = SummaryWriter("./runs")

best_mde = float('inf')

for epoch in range(epochs):
    # Training steps
    quantized_model.train()
    
    running_loss = 0.
    running_mde = 0.
    running_opa = 0.

    for i, (init_pots, wL, wR, tL, tR, label) in enumerate(tqdm(train_data_loader)):
        wL, wR, tL, tR, label = wL.to(device, dtype=torch.float), wR.to(device, dtype=torch.float), tL.to(device, dtype=torch.float), tR.to(device, dtype=torch.float), label.to(device)
        _, train_chunks = quantized_model.reformat_input_data(wL, wR, tL, tR)
        binary_chunks = (train_chunks >= 1).to(train_chunks.dtype)

        # Reset internal states + optimizers
        functional.reset_net(quantized_model)
        #safe_reset(quantized_model)
        functional.reset_net(full_precision_model)
        #safe_reset(full_precision_model)
        optimizer_m.zero_grad()
        optimizer_q.zero_grad()

        pred_s, _ = quantized_model(binary_chunks)
        pred_t, _ = full_precision_model(binary_chunks)

        loss = criterion_kd(pred_s[0], pred_t[0])
        loss.backward()
        optimizer_m.step()
        optimizer_q.step()

    
        lin_pred = pred_s[0]
        lin_label = label
        mde = MeanDepthError(lin_pred, lin_label)
        opa = OnePixelAccuracy(depth_to_disparity(lin_pred), depth_to_disparity(lin_label))

        running_loss += loss.item()
        running_mde += mde
        running_opa += opa
        
    # Log the loss, MDE, and OPA over the whole epoch
    writer.add_scalar('train/kl_loss', running_loss / len(train_data_loader), epoch)
    writer.add_scalar('train/mde', running_mde / len(train_data_loader), epoch)
    writer.add_scalar('train/opa', running_opa / len(train_data_loader), epoch)

    # Update schedulers
    scheduler_m.step()
    scheduler_q.step()

    # Validation steps
    quantized_model.eval()

    val_loss = 0.
    val_mde = 0.
    val_opa = 0.

    with torch.no_grad():
        for i, (init_pots, wL, wR, tL, tR, label) in enumerate(tqdm(val_data_loader)):    
            wL, wR, tL, tR, label = wL.to(device, dtype=torch.float), wR.to(device, dtype=torch.float), tL.to(device, dtype=torch.float), tR.to(device, dtype=torch.float), label.to(device)
            _, val_chunks = quantized_model.reformat_input_data(wL, wR, tL, tR)
            binary_chunks = (val_chunks >= 1).to(val_chunks.dtype)

            # Reset internal states
            functional.reset_net(quantized_model)
            functional.reset_net(full_precision_model)
            #safe_reset(quantized_model)
            #safe_reset(full_precision_model)
            
            pred_s, _ = quantized_model(binary_chunks)
            pred_t, _ = full_precision_model(binary_chunks)

            loss = criterion_kd(pred_s[0], pred_t[0])
            mde = MeanDepthError(pred_s[0], label)
            opa = OnePixelAccuracy(depth_to_disparity(pred_s[0]), depth_to_disparity(label))

            val_loss += loss.item()
            val_mde += mde
            val_opa += opa

    # Log the validation loss, MDE, and OPA over the whole epoch
    writer.add_scalar('val/kl_loss', val_loss / len(val_data_loader), epoch)
    writer.add_scalar('val/mde', val_mde / len(val_data_loader), epoch)
    writer.add_scalar('val/opa', val_opa / len(val_data_loader), epoch)

    # Print on standard output the epoch summary
    print(f'Epoch {epoch+1}/{epochs} - '
          f'Train Loss: {running_loss / len(train_data_loader):.4f}, '
          f'Train MDE: {running_mde / len(train_data_loader):.4f}, '
          f'Train OPA: {running_opa / len(train_data_loader):.4f} - '
          f'Val Loss: {val_loss / len(val_data_loader):.4f}, '
          f'Val MDE: {val_mde / len(val_data_loader):.4f}, '
          f'Val OPA: {val_opa / len(val_data_loader):.4f}')

   # Save the model if the performance metric (mde) is the lowest so far
    if epoch == 0 or val_mde / len(val_data_loader) < best_mde:
        best_mde = val_mde / len(val_data_loader)
        torch.save(quantized_model.state_dict(), f'best_model_quantized_stereospike.pth')


# Save the final model at the end
torch.save(quantized_model.state_dict(), f'model_quantized_StereosSpike_SQAKD_v27.pth')

writer.close()

# In[ ]:
# Load the test dataset
test_set = torch.load(cfg['datasets']['test_set'], weights_only=False)
test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=True,
                                               pin_memory=True)

# In[ ]:
def acc_eval(net, loss_module, data_loader, learned_metric = 'LIN'):
    '''
    Evaluate network accuracy as defined by the original authors of StereoSpike
        - Only for binary event frames

    Arg:
        net: network to evaluate
        loss_module: loss function definition
        data_loader: dataloader with the test dataset
        learned_metric: parameter by default is 'LIN'

    Returns:
        Print results
    '''
    # Initialize values
    running_test_loss = 0
    running_test_MDE = 0
    running_test_OPA = 0

    with torch.no_grad():
        for sample in tqdm(data_loader):

            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            warmup_chunks, test_chunks = net.reformat_input_data(warmup_chunks_left, warmup_chunks_right,
                                                                test_chunks_left, test_chunks_right)

            functional.reset_net(net)
            #safe_reset(net)

            # No warmup
            # if do_warmup:
            #     net(warmup_chunks_left, warmup_chunks_right)

            # Apply a binary mask to find all values >= 1 (True/False results)
            # then get back to the original data type.
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            # Inference
            #print(test_evframe.shape)
            pred, spks = net(test_evframe)     
            #print("max:", np.array(pred).max())
            #print("min:", np.array(pred).min())

            # Loss calculation
            loss = loss_module(pred, label, spks)
            net.detach()

            # go to linear depth to calculate MDE
            if learned_metric == 'LIN':
                lin_pred = pred[0]
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred[0])
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred[0])
            MDE = MeanDepthError(lin_pred, label)

            # go to disparity to calculate 1PA metric
            pred_disp = depth_to_disparity(lin_pred)
            gt_disp = depth_to_disparity(label)

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MDE
            running_test_OPA += OnePixelAccuracy(pred_disp, gt_disp)
            
    epoch_test_loss = running_test_loss / len(data_loader)
    epoch_test_MDE = running_test_MDE / len(data_loader)
    epoch_test_OPA = running_test_OPA / len(data_loader)
    test_epoch_summary = "Loss: {}, Mean Depth Error (m): {}, One-Pixel Accuracy: {}\n".format(
        epoch_test_loss, epoch_test_MDE, epoch_test_OPA)
    print(f'Number of samples tested: {len(data_loader)}')
    print(test_epoch_summary)

# In[ ]:
penal = False
penal_beta = 1.
loss_module = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=penal, beta=penal_beta)

print("Evaluating full precision model...")
acc_eval(full_precision_model, loss_module, test_data_loader, learned_metric='LIN')
print("Evaluating quantized model...")
acc_eval(quantized_model, loss_module, test_data_loader, learned_metric='LIN')


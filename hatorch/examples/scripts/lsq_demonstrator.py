import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from matplotlib.ticker import MultipleLocator

from hatorch.quantizers.autograd.lsq_uniform_quantizer import LsqUniformQuantizer
from hatorch.quantizers.autograd.lsq_step_driven_quantizer import LsqStepDrivenQuantizer
from hatorch.quantizers.lsq import LsqQuantizer
from hatorch.layers.quantized_conv2d import QuantConv2d
from quantized_tensor import QuantTensor

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

####################################################################################################
# Stolen from hustzxd's github : https://github.com/hustzxd/LSQuantization/blob/master/STE_LSQ.ipynb
####################################################################################################

nbits = 4
Qn = -2 ** (nbits - 1)
Qp = 2 ** (nbits - 1) - 1
# g = math.sqrt(weight.numel() * Qp)
g = 1

targets = torch.tensor([-10, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 11], dtype=torch.float).to("cpu")

Qn_rscm = torch.min(targets).item()
Qp_rscm =torch.max(targets).item()
g = 1

ws = []
wqs_int4 = []
wqs_rscm = []
alpha_grads_int4 = []
alpha_grads_rscm = []
alpha_grads_rscm_scale = []
zp_grads_rscm = []
weight_grads = []
weight_grads_rscm = []
bias_grads = []
bias_grads_rscm = []


# Compute for both quantizers
for i in range(5000):
    value = (i * 0.01) - 25
    weight = Variable(torch.Tensor([value]), requires_grad=True)
    bias = Variable(torch.Tensor([0.0]), requires_grad=True)
    alpha_int4 = torch.ones([], requires_grad=True)
    bias_int4 = torch.zeros([], requires_grad=True)

    ws.append(weight.data.item())

    # INT4 quantization
    w_q_int4 = LsqUniformQuantizer.apply(weight, alpha_int4, bias_int4, Qn, Qp) * alpha_int4 + bias_int4
    output = w_q_int4 + bias

    # Backward pass
    output.backward(retain_graph=True)
    
    # Store results
    wqs_int4.append(w_q_int4.data.item())
    alpha_grads_int4.append(alpha_int4.grad.item())
    weight_grads.append(weight.grad.item())
    bias_grads.append(bias_int4.grad.item())

    # Non-uniform RSCM quantization
    weight_rscm = Variable(torch.Tensor([value]), requires_grad=True)
    bias_rscm = Variable(torch.Tensor([0.0]), requires_grad=True)
    alpha_rscm = torch.ones([], requires_grad=True)
    zp_rscm = torch.zeros([], requires_grad=True)

    w_q_rscm = LsqStepDrivenQuantizer.apply(weight_rscm, alpha_rscm, zp_rscm, targets, Qn_rscm, Qp_rscm)
    output_rscm = w_q_rscm * alpha_rscm + zp_rscm + bias_rscm

    # Backward pass for RSCM
    output_rscm.backward()
    
    # Store RSCM results
    wqs_rscm.append(w_q_rscm.data.item())
    alpha_grads_rscm_scale.append(alpha_rscm.grad.item())
    zp_grads_rscm.append(zp_rscm.grad.item())
    weight_grads_rscm.append(weight_rscm.grad.item())
    bias_grads_rscm.append(bias_rscm.grad.item())

# Plot comparisons
fig, axes = plt.subplots(4, 2, figsize=(16, 12))

# Left column: INT4 Uniform Quantizer
axes[0, 0].plot(ws, wqs_int4, label='INT4 Uniform', color='#5e81ac', alpha=0.8, linewidth=2)
axes[0, 0].set_title('LSQ Uniform Quantizer (INT4)', fontsize=15, fontweight='bold')
axes[0, 0].set_ylabel('vhat', fontsize=15)
axes[0, 0].set_xlim(-12, 12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(ws, alpha_grads_int4, label='Scale gradient', color='#5e81ac', alpha=0.8, linewidth=2)
axes[1, 0].set_ylabel('dvhat/ds', fontsize=15)
axes[1, 0].set_xlim(-12, 12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(ws, weight_grads, label='Weight gradient', color='#bf616a', alpha=0.8, linewidth=2)
axes[2, 0].set_ylabel('dL/dw', fontsize=15)
axes[2, 0].set_xlim(-12, 12)
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[3, 0].plot(ws, bias_grads, label='Bias gradient', color='#a3be8c', alpha=0.8, linewidth=2)
axes[3, 0].set_xlabel('v', fontsize=15)
axes[3, 0].set_ylabel('dL/db', fontsize=15)
axes[3, 0].set_xlim(-12, 12)
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)

# Right column: Non-Uniform RSCM Quantizer
axes[0, 1].plot(ws, wqs_rscm, label='RSCM Non-Uniform', color='#d08770', alpha=0.8, linewidth=2)
axes[0, 1].set_title('LSQ Non-Uniform Quantizer (RSCM)', fontsize=15, fontweight='bold')
axes[0, 1].set_ylabel('vhat', fontsize=15)
axes[0, 1].set_xlim(-12, 12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(ws, alpha_grads_rscm_scale, label='Scale gradient', color='#d08770', alpha=0.8, linewidth=2)
axes[1, 1].set_ylabel('Gradients', fontsize=15)
axes[1, 1].set_xlim(-12, 12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(ws, weight_grads_rscm, label='Weight gradient', color='#bf616a', alpha=0.8, linewidth=2)
axes[2, 1].set_ylabel('dL/dw', fontsize=15)
axes[2, 1].set_xlim(-12, 12)
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

axes[3, 1].plot(ws, zp_grads_rscm, label='Bias gradient', color='#a3be8c', alpha=0.8, linewidth=2)
axes[3, 1].set_xlabel('v', fontsize=15)
axes[3, 1].set_ylabel('dL/db', fontsize=15)
axes[3, 1].set_xlim(-12, 12)
axes[3, 1].legend()
axes[3, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

plt.tight_layout()
plt.savefig("lsq_demonstrator_plots.png", dpi=150, bbox_inches='tight')
print("\nPlots saved to lsq_demonstrator_plots.png\n")
print("Left column: Uniform INT4 quantization")
print("Right column: Non-uniform RSCM quantization with targets:", targets.tolist())
# plt.savefig("lsq_comparison.pgf", format='pgf')
plt.show()

# Overlay plots: uniform vs non-uniform
fig_overlay, axes_overlay = plt.subplots(2, 1, figsize=(10, 8))

axes_overlay[0].plot(ws, wqs_int4, label='INT4', color='#3636bc', alpha=0.8, linewidth=2)
axes_overlay[0].plot(ws, wqs_rscm, label='RSCM4', color='#c85257', alpha=0.8, linewidth=2)
axes_overlay[0].set_ylabel('vhat', fontsize=15)
axes_overlay[0].set_xlim(-12, 12)
axes_overlay[0].legend()
axes_overlay[0].grid(True, alpha=0.3)

axes_overlay[1].plot(ws, alpha_grads_int4, label='INT4', color='#3636bc', alpha=0.8, linewidth=2)
axes_overlay[1].plot(ws, alpha_grads_rscm_scale, label='RSCM4', color='#c85257', alpha=0.8, linewidth=2)
axes_overlay[1].set_xlabel('v', fontsize=15)
axes_overlay[1].set_ylabel('dvhat', fontsize=15)
axes_overlay[1].set_xlim(-12, 12)
axes_overlay[1].legend()
axes_overlay[1].grid(True, alpha=0.3)

for ax in axes_overlay:
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

plt.tight_layout()
plt.savefig("examples_log/lsq_demonstrator_overlay.png", dpi=150, bbox_inches='tight')
print("Overlay plots saved to examples_log/lsq_demonstrator_overlay.png\n")
plt.show()

# Export separate PGF plots for LaTeX integration
fig_vhat, ax_vhat = plt.subplots(1, 1, figsize=(6, 2.4))
ax_vhat.plot(ws, wqs_int4, label='INT4', color='#0000ad', alpha=0.8, linewidth=2)
ax_vhat.plot(ws, wqs_rscm, label='RSCM4', color='#bc272d', alpha=0.8, linewidth=2)
ax_vhat.set_ylabel('vhat', fontsize=15)
ax_vhat.set_xlabel('v', fontsize=15, labelpad=2)
ax_vhat.set_xlim(-12, 12)
ax_vhat.xaxis.set_major_locator(MultipleLocator(2))
ax_vhat.xaxis.set_minor_locator(MultipleLocator(1))
ax_vhat.legend()
ax_vhat.grid(True, alpha=0.3)
fig_vhat.tight_layout()
fig_vhat.savefig("examples_log/lsq_vhat.pgf", format='pgf', bbox_inches='tight')
print("PGF saved to examples_log/lsq_vhat.pgf")
plt.close(fig_vhat)

fig_dvhat, ax_dvhat = plt.subplots(1, 1, figsize=(6, 2.4))
ax_dvhat.plot(ws, alpha_grads_int4, label='INT4', color='#0000ad', alpha=0.8, linewidth=2)
ax_dvhat.plot(ws, alpha_grads_rscm_scale, label='RSCM4', color='#bc272d', alpha=0.8, linewidth=2)
ax_dvhat.set_xlabel('v', fontsize=15)
ax_dvhat.set_ylabel('dvhat', fontsize=15)
ax_dvhat.set_xlim(-12, 12)
ax_dvhat.xaxis.set_major_locator(MultipleLocator(2))
ax_dvhat.xaxis.set_minor_locator(MultipleLocator(1))
ax_dvhat.legend()
ax_dvhat.grid(True, alpha=0.3)
fig_dvhat.tight_layout()
fig_dvhat.savefig("examples_log/lsq_dvhat.pgf", format='pgf', bbox_inches='tight')
print("PGF saved to examples_log/lsq_dvhat.pgf")
plt.close(fig_dvhat)

####################################################################################################
# Realistic convolution operation with gradient statistics
####################################################################################################

# print("\n" + "="*80)
# print("Realistic Convolution Operation with LSQ Quantization")
# print("="*80)
#
# # Create a simple input and convolution layer
# torch.manual_seed(42)
# input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)  # Batch=1, Channels=3, 32x32 image
# target = torch.randn(1, 16, 32, 32)  # Target output
#
# # Create quantizers for weight and activation
# weight_quantizer = LsqQuantizer(
#     bit_width=nbits,
#     signed=True,
#     symmetric=True,
#     per_channel=False,
#     is_activation=False,
#     auto_compute_g=True,
# )
#
# activation_quantizer = LsqQuantizer(
#     bit_width=nbits,
#     signed=False,
#     symmetric=False,
#     per_channel=False,
#     is_activation=True,
#     auto_compute_g=True,
# )
#
# bias_quantizer = None  # No quantization for bias in this example
#
# # Create quantized conv layer
# conv = QuantConv2d(
#     in_channels=3,
#     out_channels=16,
#     kernel_size=3,
#     stride=1,
#     padding=1,
#     bias=True,
#     weight_quantizer=weight_quantizer,
#     activation_quantizer=activation_quantizer,
#     bias_quantizer=bias_quantizer,
# )
#
# qt_tensor = QuantTensor(
#     value=input_tensor,
#     scale=torch.tensor(1.0),
#     zero_point=torch.tensor(0.0),
#     bit_width=nbits,
#     signed=False,
# )
#
# # Forward pass
# output = conv(qt_tensor)
#
# # Compute loss
# loss = nn.MSELoss()(output, target)
#
# # Backward pass
# loss.backward()
#
# # Collect gradient statistics
# print("\nGradient Statistics:")
# print("-" * 80)
#
# # Scale gradients (from weight quantizer)
# if conv.weight_quantizer.scale.grad is not None:
#     scale_grad = conv.weight_quantizer.scale.grad
#     print(f"Weight Scale Gradient:")
#     print(f"  Max:  {scale_grad.max().item():.6e}")
#     print(f"  Mean: {scale_grad.mean().item():.6e}")
#     print(f"  Min:  {scale_grad.min().item():.6e}")
# else:
#     print("Weight Scale Gradient: None")
#
# # Weight gradients
# if conv.weight.grad is not None:
#     weight_grad = conv.weight.grad
#     print(f"\nWeight Gradient:")
#     print(f"  Max:  {weight_grad.max().item():.6e}")
#     print(f"  Mean: {weight_grad.mean().item():.6e}")
#     print(f"  Min:  {weight_grad.min().item():.6e}")
# else:
#     print("\nWeight Gradient: None")
#
# # Bias gradients
# if conv.bias is not None and conv.bias.grad is not None:
#     bias_grad = conv.bias.grad
#     print(f"\nBias Gradient:")
#     print(f"  Max:  {bias_grad.max().item():.6e}")
#     print(f"  Mean: {bias_grad.mean().item():.6e}")
#     print(f"  Min:  {bias_grad.min().item():.6e}")
# else:
#     print("\nBias Gradient: None")
#
# # Bias scale gradients (from bias quantizer)
# if bias_quantizer:
#     if conv.bias_quantizer.scale.grad is not None:
#         bias_scale_grad = conv.bias_quantizer.scale.grad
#         print(f"\nBias Scale Gradient:")
#         print(f"  Max:  {bias_scale_grad.max().item():.6e}")
#         print(f"  Mean: {bias_scale_grad.mean().item():.6e}")
#         print(f"  Min:  {bias_scale_grad.min().item():.6e}")
#     else:
#         print("\nBias Scale Gradient: None")
#
# # Activation scale gradients
# if conv.activation_quantizer.scale.grad is not None:
#     act_scale_grad = conv.activation_quantizer.scale.grad
#     print(f"\nActivation Scale Gradient:")
#     print(f"  Max:  {act_scale_grad.max().item():.6e}")
#     print(f"  Mean: {act_scale_grad.mean().item():.6e}")
#     print(f"  Min:  {act_scale_grad.min().item():.6e}")
# else:
#     print("\nActivation Scale Gradient: None")
#
# # Activation zero-point gradients
# if conv.activation_quantizer.zero_point is not None and conv.activation_quantizer.zero_point.grad is not None:
#     act_zp_grad = conv.activation_quantizer.zero_point.grad
#     print(f"\nActivation Zero-Point Gradient:")
#     print(f"  Max:  {act_zp_grad.max().item():.6e}")
#     print(f"  Mean: {act_zp_grad.mean().item():.6e}")
#     print(f"  Min:  {act_zp_grad.min().item():.6e}")
# else:
#     print("\nActivation Zero-Point Gradient: None")
#
# print("\n" + "="*80)

# RIFE v4.25 (IFNet) in PyTorch, trainable. The same computation as the extension's WebGPU
# implementation (verified against it to about 3e-5 px of flow); the architecture is that of
# Practical-RIFE (MIT, see THIRD_PARTY_NOTICES.md). Parameters keep the checkpoint's names, so a
# fine-tuned model is saved as a checkpoint that tools/export_weights.py converts as usual.
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from export_weights import load_state_dict  # noqa: E402  (restricted unpickler: tensors only)

SCALES = [16, 8, 4, 2, 1]
MODULO = 64  # input sides must be multiples of this


def lrelu(x):
    return F.leaky_relu(x, 0.2)


class IFNet(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        sd = load_state_dict(checkpoint)
        self.names = list(sd.keys())
        trained = [k for k in sd if k.startswith('encode.') or k.startswith('block')]
        # (the checkpoint's teacher and timestep branches are not used at inference: kept as saved)
        self.extra = OrderedDict((k, torch.from_numpy(np.array(v, dtype=np.float32))) for k, v in sd.items() if k not in trained)
        self.p = nn.ParameterDict({k.replace('.', '__'): nn.Parameter(torch.from_numpy(np.array(sd[k], dtype=np.float32))) for k in trained})
        self.grids = {}

    def w(self, name):
        return self.p[name.replace('.', '__')]

    def head(self, x):
        p = 'encode.'
        x = lrelu(F.conv2d(x, self.w(p + 'cnn0.weight'), self.w(p + 'cnn0.bias'), stride=2, padding=1))
        x = lrelu(F.conv2d(x, self.w(p + 'cnn1.weight'), self.w(p + 'cnn1.bias'), padding=1))
        x = lrelu(F.conv2d(x, self.w(p + 'cnn2.weight'), self.w(p + 'cnn2.bias'), padding=1))
        return F.conv_transpose2d(x, self.w(p + 'cnn3.weight'), self.w(p + 'cnn3.bias'), stride=2, padding=1)

    def block(self, i, x, flow, scale):
        p = f'block{i}.'
        x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear')
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode='bilinear') / scale
            x = torch.cat((x, flow), 1)
        x = lrelu(F.conv2d(x, self.w(p + 'conv0.0.0.weight'), self.w(p + 'conv0.0.0.bias'), stride=2, padding=1))
        x = lrelu(F.conv2d(x, self.w(p + 'conv0.1.0.weight'), self.w(p + 'conv0.1.0.bias'), stride=2, padding=1))
        for j in range(8):
            q = p + f'convblock.{j}.'
            x = lrelu(F.conv2d(x, self.w(q + 'conv.weight'), self.w(q + 'conv.bias'), padding=1) * self.w(q + 'beta') + x)
        tmp = F.pixel_shuffle(F.conv_transpose2d(x, self.w(p + 'lastconv.0.weight'), self.w(p + 'lastconv.0.bias'), stride=2, padding=1), 2)
        tmp = F.interpolate(tmp, scale_factor=scale, mode='bilinear')
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    def warp(self, img, flow):
        n, c, h, w = img.shape
        key = (h, w, img.device)
        if key not in self.grids:
            gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=img.device), torch.arange(w, dtype=torch.float32, device=img.device), indexing='ij')
            self.grids[key] = (gx, gy)
        gx, gy = self.grids[key]
        x = (gx + flow[:, 0]) / ((w - 1) / 2) - 1
        y = (gy + flow[:, 1]) / ((h - 1) / 2) - 1
        return F.grid_sample(img, torch.stack((x, y), -1).to(img.dtype), mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, img0, img1, t):
        """img0, img1: (N, 3, H, W) in 0..1; t: (N,) timesteps. Returns flow (N, 4, H, W), mask, frame."""
        f0, f1 = self.head(img0), self.head(img1)
        tt = t.view(-1, 1, 1, 1).to(img0.dtype).expand(-1, 1, img0.shape[2], img0.shape[3])
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, s in enumerate(SCALES):
            if flow is None:
                flow, mask, feat = self.block(i, torch.cat((img0, img1, f0, f1, tt), 1), None, s)
            else:
                wf0, wf1 = self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4])
                fd, mask, feat = self.block(i, torch.cat((w0, w1, wf0, wf1, tt, mask, feat), 1), flow, s)
                flow = flow + fd
            w0, w1 = self.warp(img0, flow[:, :2]), self.warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return flow, m, w0 * m + w1 * (1 - m)

    def checkpoint(self):
        """The full state dict (the checkpoint's names and order), for torch.save()."""
        out = OrderedDict()
        for k in self.names:
            out[k] = self.p[k.replace('.', '__')].detach().float().cpu() if k.replace('.', '__') in self.p else self.extra[k]
        return out

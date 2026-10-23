# Fine-tune RIFE v4.25 on synthetic triplets (synth.py), optionally mixed with real video frames.
#   python train.py                      (defaults: 20000 steps, batch 16, 256 x 256 crops)
#   python train.py --real frames_dir    (also triplets of consecutive frames from PNG sequences)
# Writes out/rife-v4.25-ft.pkl (the best checkpoint on the validation set) and out/log.txt.
import argparse
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import IFNet, MODULO
from synth import make_batch

BASE_URL = 'https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl'
HERE = Path(__file__).resolve().parent


def lap_loss(x, y, levels=5):
    """L1 over a Laplacian pyramid (fine detail and coarse structure alike)."""
    loss = 0
    for _ in range(levels):
        dx, dy = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
        ux, uy = F.interpolate(dx, scale_factor=2, mode='bilinear'), F.interpolate(dy, scale_factor=2, mode='bilinear')
        loss = loss + ((x - ux) - (y - uy)).abs().mean()
        x, y = dx, dy
    return loss + (x - y).abs().mean()


def psnr(x, y):
    return float((-10 * torch.log10(((x - y) ** 2).mean((1, 2, 3)) + 1e-10)).mean())


class RealFrames:
    """Triplets of consecutive frames (k, k + 1, k + 2) from folders of PNG sequences (one folder per
    clip); the middle frame is the target at t = 0.5. Frames are loaded at half size."""

    def __init__(self, root, size):
        from PIL import Image
        self.Image = Image
        self.size = size
        self.clips = []
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            files = sorted(d.glob('*.png'))
            if len(files) >= 3:
                self.clips.append(files)
        if not self.clips:
            raise SystemExit(f'no PNG sequences found under {root}')
        n = sum(len(c) for c in self.clips)
        print(f'real frames: {len(self.clips)} clips, {n} frames')
        self.cache = {}

    def load(self, f):
        if f not in self.cache:
            im = self.Image.open(f).convert('RGB')
            im = im.resize((im.width // 2, im.height // 2), self.Image.BILINEAR)
            self.cache[f] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
            if len(self.cache) > 3000:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[f]

    def batch(self, n, dev, rng):
        A, T, B = [], [], []
        while len(A) < n:
            clip = rng.choice(self.clips)
            k = rng.randrange(len(clip) - 2)
            fr = [self.load(f) for f in clip[k:k + 3]]
            h, w = fr[0].shape[1:]
            if h < self.size or w < self.size:
                continue
            y, x = rng.randrange(h - self.size + 1), rng.randrange(w - self.size + 1)
            a, t, b = [f[:, y:y + self.size, x:x + self.size].float() / 255 for f in fr]
            if (a - b).abs().mean() < 0.004 or (a - b).abs().mean() > 0.25:  # repeated drawing, or a cut
                continue
            A.append(a), T.append(t), B.append(b)
        return torch.stack(A).to(dev), torch.stack(T).to(dev), torch.stack(B).to(dev), torch.full((n,), 0.5, device=dev)


def evaluate(model, val, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for A, T, B, t in val:
            out.append(psnr(model(A.to(dev), B.to(dev), t.to(dev))[2].clamp(0, 1), T.to(dev)))
    model.train()
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=str(HERE / 'flownet_v4.25.pkl'), help='RIFE v4.25 checkpoint (downloaded if missing)')
    ap.add_argument('--out', default=str(HERE / 'out'))
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--keep', type=float, default=1e-2, help='pull towards the original weights (keeps what it already does well)')
    ap.add_argument('--real', default=None, help='folder of PNG sequences (optional)')
    ap.add_argument('--real-share', type=float, default=0.3)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    assert a.size % MODULO == 0, f'--size must be a multiple of {MODULO}'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        print('warning: no CUDA GPU found, training on the CPU (very slow)')
    else:
        print('GPU:', torch.cuda.get_device_name(0))
    base = Path(a.base)
    if not base.exists():
        print('downloading', BASE_URL)
        urllib.request.urlretrieve(BASE_URL, base)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.txt', 'a', encoding='utf-8')

    def say(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    model = IFNet(base).to(dev)
    ref = [p.detach().clone() for p in model.p.values()]
    real = RealFrames(a.real, a.size) if a.real else None
    rng = random.Random(a.seed)
    # validation: fixed synthetic triplets (never trained on: other seeds), and real ones if given
    val = [make_batch(8, a.size, a.size, dev, seed=10_000_000 + k) for k in range(8)]
    val_real = [real.batch(8, dev, random.Random(424242 + k)) for k in range(4)] if real else []
    base_syn = evaluate(model, val, dev)
    base_real = evaluate(model, val_real, dev) if real else None
    say(f'original model: synthetic {base_syn:.2f} dB' + (f', real {base_real:.2f} dB' if real else ''))
    best = -1e9
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
    start = 1
    resume = out / 'resume.pt'
    if resume.exists():
        st = torch.load(resume, map_location=dev, weights_only=True)
        if st['steps'] == a.steps:
            model.load_state_dict(st['model'])
            opt.load_state_dict(st['opt'])
            sched.load_state_dict(st['sched'])
            best, start = st['best'], st['step'] + 1
            say(f'resuming at step {start}')
    t0 = time.time()
    for step in range(start, a.steps + 1):
        if real and rng.random() < a.real_share:
            A, T, B, t = real.batch(a.batch, dev, rng)
        else:
            A, T, B, t = make_batch(a.batch, a.size, a.size, dev, seed=a.seed * 1_000_003 + step)
        if rng.random() < 0.5:  # the same pair the other way round
            A, B, t = B, A, 1 - t
        if rng.random() < 0.5:
            A, T, B = A.flip(3), T.flip(3), B.flip(3)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            pred = model(A, B, t)[2]
        loss = lap_loss(pred.float(), T)
        keep = sum(((p - r) ** 2).sum() for p, r in zip(model.p.values(), ref))
        total = loss + a.keep * keep
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step == start + 29:
            per = (time.time() - t0) / 30
            say(f'about {per:.2f} s per step: {a.steps} steps take about {per * a.steps / 3600:.1f} h (plus evaluations)')
        if step % 100 == 0:
            say(f'step {step}/{a.steps} loss {float(loss.detach()):.4f} keep {float(keep.detach()):.3e} ({(time.time() - t0) / 60:.0f} min)')
        if step % a.eval_every == 0 or step == a.steps:
            syn = evaluate(model, val, dev)
            rl = evaluate(model, val_real, dev) if real else None
            score = syn + (rl if rl is not None else 0)
            msg = f'  validation: synthetic {syn:.2f} dB ({syn - base_syn:+.2f})' + (f', real {rl:.2f} dB ({rl - base_real:+.2f})' if real else '')
            if score > best:
                best = score
                torch.save(model.checkpoint(), out / 'rife-v4.25-ft.pkl')
                msg += '  -> saved'
            say(msg)
        if step % 200 == 0:  # (to resume after an interruption: run the same command again)
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'best': best, 'steps': a.steps}, resume)
    torch.save(model.checkpoint(), out / 'rife-v4.25-ft-last.pkl')
    if resume.exists():
        resume.unlink()
    say(f'done in {(time.time() - t0) / 3600:.1f} h. Send {out / "rife-v4.25-ft.pkl"} (about 25 MB).')


if __name__ == '__main__':
    main()

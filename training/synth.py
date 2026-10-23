# Synthetic training triplets, generated on the GPU on the fly: a background and up to four objects
# with procedural textures (noise, city windows, stripes / checks / bricks, flat cel shading with
# outlines, pixel art, sky gradients), each moving with its own translation, zoom (approaching /
# receding) and rotation. Frames are rendered at t = 0 (A), t (the target) and 1 (B) with 2x2
# supersampling, then lightly blurred / noised / quantised to 8 bits like video.
# The mix leans towards the motions RIFE handles worst: large motion over repetitive patterns, and
# objects that change size quickly.
import math

import torch
import torch.nn.functional as F

SS = 2  # supersampling per axis


class Rng:
    def __init__(self, gen):
        self.g = gen

    def u(self, a=0.0, b=1.0):
        return a + (b - a) * float(torch.rand((), generator=self.g))

    def i(self, a, b):  # a..b inclusive
        return int(torch.randint(a, b + 1, (), generator=self.g))

    def p(self, prob):
        return self.u() < prob

    def pick(self, items, weights=None):
        w = torch.tensor(weights if weights else [1.0] * len(items))
        return items[int(torch.multinomial(w, 1, generator=self.g))]

    def color(self, lo=0.0, hi=1.0):
        return [self.u(lo, hi) for _ in range(3)]


def periodic_sample(grid, u, v):
    """Bilinear sample of a (C, n, n) grid at (u, v) in grid cells, wrapping around."""
    c, n, _ = grid.shape
    g = F.pad(grid[None], (0, 1, 0, 1), mode='circular')
    # (align_corners=True over the n + 1 padded samples: normalised -1..1 -> index 0..n)
    x = torch.remainder(u, n) / n * 2 - 1
    y = torch.remainder(v, n) / n * 2 - 1
    return F.grid_sample(g, torch.stack((x, y), -1)[None], mode='bilinear', align_corners=True)[0]


def hash2(a, b):
    return torch.frac(torch.sin(a * 12.9898 + b * 78.233) * 43758.5453)


def make_texture(r, dev):
    """A procedural texture: f(u, v) -> (3, H, W) colours, (u, v) in pixels."""
    kind = r.pick(['noise', 'city', 'stripes', 'checks', 'bricks', 'cel', 'pixel', 'sky'], [2, 2, 1.5, 1, 1, 2.5, 1, 0.5])
    c0, c1 = torch.tensor(r.color(), device=dev).view(3, 1, 1), torch.tensor(r.color(), device=dev).view(3, 1, 1)
    if kind == 'noise':
        # smooth multi-octave noise between two colours (plus a little colour variation)
        grids = [(torch.rand(1, 32, 32, device=dev), r.u(10, 60) * k) for k in (1, 0.5, 0.25)]
        tint = torch.rand(3, 16, 16, device=dev)
        amp = [1.0, 0.5, 0.25]
        contrast = r.u(0.5, 1.5)

        def f(u, v):
            s = sum(a * periodic_sample(g, u / sc, v / sc) for (g, sc), a in zip(grids, amp)) / sum(amp)
            s = ((s - 0.5) * contrast * 2 + 0.5).clamp(0, 1)
            return (c0 * (1 - s) + c1 * s + 0.15 * (periodic_sample(tint, u / 80, v / 80) - 0.5)).clamp(0, 1)
        return f
    if kind == 'city':
        px, py = r.u(5, 18), r.u(6, 20)
        wx, wy = r.u(0.35, 0.7), r.u(0.35, 0.7)
        lit, dark = torch.tensor(r.color(0.6, 1.0), device=dev).view(3, 1, 1), torch.tensor(r.color(0.0, 0.3), device=dev).view(3, 1, 1)
        wall = c0 * 0.6
        bw = r.u(60, 160)

        def f(u, v):
            iu, iv = torch.floor(u / px), torch.floor(v / py)
            win = (torch.frac(u / px) < wx) & (torch.frac(v / py) < wy)
            on = hash2(iu, iv) > 0.45
            b = torch.floor(u / bw)
            shade = 0.7 + 0.3 * hash2(b, b * 0.37)
            col = torch.where(win, torch.where(on, lit, dark), wall * shade)
            return col
        return f
    if kind in ('stripes', 'checks', 'bricks'):
        period = r.u(4, 24)
        ang = r.u(0, math.pi)
        duty = r.u(0.3, 0.7)

        def f(u, v):
            if kind == 'stripes':
                s = torch.frac((u * math.cos(ang) + v * math.sin(ang)) / period) < duty
            elif kind == 'checks':
                s = (torch.floor(u / period) + torch.floor(v / period)) % 2 == 0
            else:
                row = torch.floor(v / period)
                uu = u + (row % 2) * period
                s = (torch.frac(uu / (2 * period)) > 0.06) & (torch.frac(v / period) > 0.12)
            return torch.where(s, c0, c1)
        return f
    if kind == 'cel':
        grid = torch.rand(1, 16, 16, device=dev)
        sc = r.u(20, 80)
        levels = r.i(2, 4)
        pal = torch.tensor([r.color() for _ in range(levels + 1)], device=dev)
        line = torch.tensor(r.color(0.0, 0.15), device=dev).view(3, 1, 1)

        def lev(u, v):
            return torch.floor(periodic_sample(grid, u / sc, v / sc)[0] * levels * 0.999 * 1.6 - 0.3 * levels).clamp(0, levels)

        def f(u, v):
            l0 = lev(u, v)
            edge = (lev(u + 1.2, v) != l0) | (lev(u, v + 1.2) != l0)
            col = pal[l0.long()].permute(2, 0, 1)
            return torch.where(edge[None], line, col)
        return f
    if kind == 'pixel':
        # pixel art: blobby shapes in a small palette at a coarse cell size, nearest-neighbour
        cell = r.i(3, 8)
        n = 32
        pal = torch.tensor([r.color() for _ in range(5)], device=dev)
        sm = F.interpolate(torch.rand(1, 1, 6, 6, device=dev), size=(n, n), mode='bicubic', align_corners=False)[0, 0]
        idx = (sm.clamp(0, 0.999) * 5).long()
        img = pal[idx].permute(2, 0, 1)

        def f(u, v):
            iu = torch.remainder(torch.floor(u / cell), n).long()
            iv = torch.remainder(torch.floor(v / cell), n).long()
            return img[:, iv, iu]
        return f
    grid = torch.rand(3, 8, 8, device=dev)

    def f(u, v):  # sky: vertical gradient + soft clouds
        g = (v / 300.0).clamp(-1, 1)[None] * 0.5 + 0.5
        return (c0 * (1 - g) + c1 * g) * 0.8 + 0.2 * periodic_sample(grid, u / 90, v / 90)
    return f


def speed(r, big):
    """Motion over the pair (px): mostly modest, sometimes large (big: the scale of 'large')."""
    k = r.pick(['small', 'mid', 'large'], [0.45, 0.35, 0.2])
    mag = r.u(0, 12) if k == 'small' else r.u(12, big * 0.5) if k == 'mid' else r.u(big * 0.5, big)
    a = r.u(0, 2 * math.pi)
    return mag * math.cos(a), mag * math.sin(a)


def render_layer(tex, mask_fn, xs, ys, c, s, th, deform=None):
    """Colour and alpha of a layer whose local origin is at screen point c, zoomed s, rotated th;
    deform(u, v) -> (u, v) bends the shape (non-rigid motion)."""
    dx, dy = xs - c[0], ys - c[1]
    ct, st = math.cos(-th), math.sin(-th)
    u = (dx * ct - dy * st) / s
    v = (dx * st + dy * ct) / s
    if deform is not None:
        u, v = deform(u, v)
    col = tex(u, v)
    if mask_fn is None:
        return col, None
    a, outline = mask_fn(u, v, s)
    return col, (a, outline)


def make_mask(r):
    shape = r.pick(['ellipse', 'rect', 'star'])
    rx, ry = r.u(14, 90), r.u(14, 90)
    k, amp = r.i(3, 7), r.u(0.1, 0.4)
    ow = r.u(1.0, 3.5) if r.p(0.6) else 0.0  # outline width (screen px)

    def m(u, v, s):
        if shape == 'ellipse':
            d = torch.sqrt((u / rx) ** 2 + (v / ry) ** 2)
            rad = min(rx, ry)
        elif shape == 'rect':
            d = torch.maximum((u / rx).abs(), (v / ry).abs())
            rad = min(rx, ry)
        else:
            ang = torch.atan2(v, u)
            rr = rx * (1 + amp * torch.sin(k * ang))
            d = torch.sqrt(u ** 2 + v ** 2) / rr
            rad = rx
        px = 1.0 / (rad * s)  # one screen px in units of d
        alpha = ((1 - d) / px + 0.5).clamp(0, 1)
        outline = ((1 - d) / px < ow) & (alpha > 0) if ow > 0 else None
        return alpha, outline
    return m


def make_bend(r, dev):
    """Non-rigid motion of an object (limbs swinging, cloth, faces): a smooth random displacement
    field plus a sinusoidal sway, both growing with t (px in the object's own coordinates)."""
    grid = torch.rand(2, 8, 8, device=dev)
    sc = r.u(15, 60)
    amp = r.u(2, 14)
    sway, wl, ph = r.u(0, 12), r.u(20, 80), r.u(0, 2 * math.pi)

    def f(u, v, t):
        d = (periodic_sample(grid, u / sc, v / sc) - 0.5) * 2 * amp * t
        return u + d[0] + sway * t * torch.sin(v / wl + ph), v + d[1]
    return f


def make_scene(r, H, W, dev):
    """One scene: a function render(t) -> (3, H, W)."""
    bg = make_texture(r, dev)
    big = 0.45 * max(H, W)
    bgv = speed(r, big) if r.p(0.6) else (0.0, 0.0)  # (a still background with small motion in front is common)
    bgz = math.exp(r.u(-0.2, 0.2)) if r.p(0.25) else 1.0
    bgr = r.u(-0.12, 0.12) if r.p(0.12) else 0.0
    bgo = (r.u(-500, 500), r.u(-500, 500))
    acc = r.u(-0.2, 0.2) if r.p(0.3) else 0.0
    objs = []
    for _ in range(r.pick([0, 1, 2, 3, 4], [0.15, 0.3, 0.25, 0.2, 0.1])):
        approach = r.p(0.35)
        objs.append({
            'tex': make_texture(r, dev),
            'mask': make_mask(r),
            'line': torch.tensor(r.color(0.0, 0.15), device=dev).view(3, 1, 1),
            'c0': (r.u(-0.1, 1.1) * W, r.u(-0.1, 1.1) * H),
            'v': speed(r, big),
            's0': r.u(0.6, 1.4),
            'z': math.exp(r.u(-0.7, 0.7)) if approach else (math.exp(r.u(-0.1, 0.1)) if r.p(0.3) else 1.0),
            'w': r.u(-0.4, 0.4) if r.p(0.2) else 0.0,
            'acc': r.u(-0.25, 0.25) if r.p(0.3) else 0.0,
            'bend': make_bend(r, dev) if r.p(0.4) else None,
        })
    gain = r.u(-0.35, 0.35) if r.p(0.1) else 0.0
    iy, ix = torch.meshgrid(torch.arange(H * SS, device=dev, dtype=torch.float32), torch.arange(W * SS, device=dev, dtype=torch.float32), indexing='ij')
    xs = (ix + 0.5) / SS - 0.5
    ys = (iy + 0.5) / SS - 0.5

    def render(t):
        # background: the screen centre moves along -v (camera pan), zoom and rotation about the centre
        k = t + acc * t * (t - 1)
        c = (W / 2 + bgo[0] * 0 - bgv[0] * k, H / 2 - bgv[1] * k)
        col, _ = render_layer(lambda u, v: bg(u + bgo[0], v + bgo[1]), None, xs, ys, c, bgz ** t, bgr * t)
        for o in objs:
            k = t + o['acc'] * t * (t - 1)
            c = (o['c0'][0] + o['v'][0] * k, o['c0'][1] + o['v'][1] * k)
            bend = o['bend']
            oc, (a, outline) = render_layer(o['tex'], o['mask'], xs, ys, c, o['s0'] * o['z'] ** t, o['w'] * t, (lambda u, v, b=bend, t=t: b(u, v, t)) if bend else None)
            if outline is not None:
                oc = torch.where(outline[None], o['line'], oc)
            col = col * (1 - a) + oc * a
        col = col * (1 + gain * t)
        return F.avg_pool2d(col[None], SS)[0].clamp(0, 1)
    return render


def degrade(x, r):
    """Video-like: slight blur sometimes, noise, 8-bit quantisation (same for all three frames)."""
    blur = r.u(0.3, 0.9) if r.p(0.3) else 0.0
    sigma = r.u(0, 2.0) / 255

    def f(img, seed):
        if blur > 0:
            k = torch.tensor([math.exp(-(i * i) / (2 * blur * blur)) for i in (-1, 0, 1)], device=img.device)
            k = k / k.sum()
            img = F.conv2d(F.pad(img[None], (1, 1, 1, 1), mode='replicate'), (k[:, None] * k[None, :]).expand(3, 1, 3, 3), groups=3)[0]
        g = torch.Generator(device='cpu').manual_seed(seed)
        img = img + torch.randn(img.shape, generator=g).to(img.device) * sigma
        return torch.round(img.clamp(0, 1) * 255) / 255
    return f


def make_batch(n, H, W, dev, seed, t_mid=0.8):
    """n triplets: (A, target, B, t) with A at t = 0, B at t = 1, target at t (0.5 with probability t_mid)."""
    gen = torch.Generator().manual_seed(seed)
    r = Rng(gen)
    A, T, B, ts = [], [], [], []
    for k in range(n):
        torch.manual_seed(seed * 1000 + k)  # (textures draw from the global generator)
        scene = make_scene(r, H, W, dev)
        t = 0.5 if r.p(t_mid) else r.u(0.15, 0.85)
        d = degrade(None, r)
        s0 = seed * 7919 + k * 3
        A.append(d(scene(0.0), s0))
        T.append(d(scene(t), s0 + 1))
        B.append(d(scene(1.0), s0 + 2))
        ts.append(t)
    return torch.stack(A), torch.stack(T), torch.stack(B), torch.tensor(ts, device=dev)

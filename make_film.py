#!/usr/bin/env python3
"""Render a short family film (vertical 1080x1920) from the two photos.

Requires: pillow (with raqm), numpy, imageio-ffmpeg
Usage:    python3 make_film.py            -> writes family_film.mp4
"""
import math
import random
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).parent
W, H, FPS = 1080, 1920, 30
OUT = ROOT / "family_film.mp4"
AUDIO = ROOT / "build_music.wav"

FONT_TITLE = str(ROOT / "assets/fonts/Amiri-Bold.ttf")
FONT_TEXT = str(ROOT / "assets/fonts/ReemKufi.ttf")
GOLD = (240, 200, 120)

ORIG = Image.open(ROOT / "photos/family_original.jpg").convert("RGB")
ART = Image.open(ROOT / "photos/family_art.png").convert("RGB")
# Warm, slightly nostalgic grade for the real photo
ORIG = ImageEnhance.Color(ORIG).enhance(1.15)
ORIG = ImageEnhance.Contrast(ORIG).enhance(1.05)

# Face centres as fractions of each photo (x, y)
FACES_ART = {
    "baby": (0.71, 0.73),
    "left": (0.19, 0.43),
    "girl": (0.46, 0.39),
    "big": (0.74, 0.26),
    "dad": (0.40, 0.66),
}
FACES_ORIG = {
    "baby": (0.67, 0.84),
    "left": (0.15, 0.55),
    "girl": (0.44, 0.45),
    "big": (0.71, 0.37),
    "dad": (0.37, 0.80),
}

# ---------------------------------------------------------------- helpers

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def crop_view(img, cx, cy, zoom):
    """9:16 view centred on (cx, cy) fractions; zoom=1 means tallest crop."""
    iw, ih = img.size
    ch = min(ih, iw * 16 / 9) / zoom
    cw = ch * 9 / 16
    x = min(max(cx * iw - cw / 2, 0), iw - cw)
    y = min(max(cy * ih - ch / 2, 0), ih - ch)
    return img.transform((W, H), Image.Transform.EXTENT, (x, y, x + cw, y + ch),
                         Image.Resampling.BICUBIC)


_bg_cache = {}


def framed(img, zoom=1.0, border=True):
    """Whole photo centred over a blurred, darkened copy of itself."""
    key = id(img)
    if key not in _bg_cache:
        bg = crop_view(img, 0.5, 0.5, 1.0).filter(ImageFilter.GaussianBlur(40))
        _bg_cache[key] = ImageEnhance.Brightness(bg).enhance(0.45)
    frame = _bg_cache[key].copy()
    iw, ih = img.size
    fw = int(W * 0.88 * zoom)
    fh = int(fw * ih / iw)
    photo = img.resize((fw, fh), Image.Resampling.BICUBIC)
    x, y = (W - fw) // 2, (H - fh) // 2 - 60
    if border:
        d = ImageDraw.Draw(frame)
        d.rectangle((x - 14, y - 14, x + fw + 14, y + fh + 14), fill=(250, 245, 235))
    frame.paste(photo, (x, y))
    return frame


def vignette_mask():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    v = np.clip(1.15 - d * 0.9, 0.35, 1.0)
    return v[..., None].astype(np.float32)


VIGNETTE = vignette_mask()


def text_layer(frame, text, y, size, alpha, font=FONT_TEXT, color=(255, 255, 255)):
    if alpha <= 0:
        return frame
    f = ImageFont.truetype(font, size)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), text, font=f, direction="rtl", language="ar")
    tw = bbox[2] - bbox[0]
    x = (W - tw) / 2 - bbox[0]
    a = int(255 * alpha)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((x + 3, y + 5), text, font=f, fill=(0, 0, 0, int(a * 0.8)),
                                direction="rtl", language="ar")
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    d.text((x, y), text, font=f, fill=color + (a,), direction="rtl", language="ar")
    base = frame.convert("RGBA")
    base = Image.alpha_composite(base, shadow)
    base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


random.seed(7)
SPARKS = [(random.random(), random.random(), random.uniform(0.3, 1.0),
           random.uniform(2, 6), random.random() * 6.28) for _ in range(70)]


def sparkles(frame, t, strength=1.0):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for sx, sy, speed, r, ph in SPARKS:
        x = (sx * W + math.sin(t * 0.7 + ph) * 40) % W
        y = (sy * H - t * speed * 90) % H
        a = int(255 * strength * (0.5 + 0.5 * math.sin(t * 3 + ph)))
        d.ellipse((x - r, y - r, x + r, y + r), fill=GOLD + (a,))
    layer = layer.filter(ImageFilter.GaussianBlur(1.5))
    return Image.alpha_composite(frame.convert("RGBA"), layer).convert("RGB")


def finish(frame, fade=1.0):
    arr = np.asarray(frame, dtype=np.float32) * VIGNETTE * fade
    return np.clip(arr, 0, 255).astype(np.uint8)


def fade_io(t, dur, fin=0.6, fout=0.6):
    return ease(t / fin) * ease((dur - t) / fout)


# ---------------------------------------------------------------- scenes
# Each scene: (duration, fn(t) -> PIL image, fade-in, fade-out of the text)

def s_intro(t, dur):
    frame = Image.new("RGB", (W, H), (8, 6, 4))
    frame = sparkles(frame, t, ease(t / 1.5))
    frame = text_layer(frame, "حكاية عائلة", 700, 150, fade_io(t, dur, 1.2, 0.8),
                       font=FONT_TITLE, color=GOLD)
    frame = text_layer(frame, "أجمل ما نملك… نحن", 930, 64, fade_io(t - 1.2, dur - 1.2, 1.0, 0.8))
    return frame


def s_original(t, dur):
    p = ease(t / dur)
    frame = framed(ORIG, zoom=1.0 + 0.06 * p)
    frame = text_layer(frame, "لحظة عادية في مساء عادي…", 1640, 62, fade_io(t - 0.5, dur - 0.5))
    return frame


def s_flip(t, dur):
    """Card-flip from the real photo to the art version, with a warm flash."""
    p = ease(t / dur)
    src = ORIG if p < 0.5 else ART
    squash = abs(math.cos(p * math.pi))
    full = framed(src, zoom=1.06 if src is ORIG else 1.0)
    bg = framed(src, zoom=1.0)
    w = max(int(W * squash), 2)
    card = full.resize((w, H))
    bg.paste(card, ((W - w) // 2, 0))
    flash = max(0.0, 1 - abs(p - 0.5) * 6)
    return Image.blend(bg, Image.new("RGB", (W, H), (255, 240, 210)), flash * 0.8)


def s_art(t, dur):
    p = ease(t / dur)
    frame = framed(ART, zoom=1.0 + 0.05 * p)
    frame = sparkles(frame, t, 0.6)
    frame = text_layer(frame, "صارت ذكرى لا تُنسى", 1640, 62, fade_io(t - 0.3, dur - 0.3))
    return frame


CLOSEUPS = [
    ("baby", "ضحكة تُنير البيت", ART),
    ("left", "براءة وأحلام كبيرة", ART),
    ("girl", "خجل الأميرة الصغيرة", ART),
    ("big", "الأخ الكبير… السند", ART),
    ("dad", "وأب قلبه يتّسع للجميع", ART),
]


def make_closeup(key, caption, img):
    faces = FACES_ART if img is ART else FACES_ORIG

    def scene(t, dur):
        p = ease(t / dur)
        cx, cy = faces[key]
        z = 1.6 + 0.35 * p
        frame = crop_view(img, cx + 0.02 * (p - 0.5), cy + 0.04, z)
        frame = text_layer(frame, caption, 1620, 70, fade_io(t - 0.3, dur - 0.3, 0.5, 0.5))
        return frame
    return scene


def s_then_now(t, dur):
    """Real vs. art side by side, sliding divider."""
    a = crop_view(ORIG, 0.46, 0.62, 1.0)
    b = crop_view(ART, 0.46, 0.55, 1.0)
    p = ease(t / dur)
    split = int(W * (0.85 - 0.7 * p))
    frame = a.copy()
    frame.paste(b.crop((split, 0, W, H)), (split, 0))
    d = ImageDraw.Draw(frame)
    d.rectangle((split - 3, 0, split + 3, H), fill=(255, 255, 255))
    frame = text_layer(frame, "الحقيقة… والخيال", 200, 72, fade_io(t, dur), color=GOLD)
    return frame


def s_outro(t, dur):
    p = ease(t / dur)
    frame = framed(ART, zoom=1.08 - 0.08 * p)
    frame = ImageEnhance.Brightness(frame).enhance(1 - 0.55 * ease((t - 1.5) / 2))
    frame = sparkles(frame, t, 1.0)
    frame = text_layer(frame, "البيت ليس جدرانًا…", 650, 78, fade_io(t - 1.5, dur - 1.5, 1.0, 1.0))
    frame = text_layer(frame, "البيت هم", 800, 160, fade_io(t - 2.8, dur - 2.8, 1.0, 1.0),
                       font=FONT_TITLE, color=GOLD)
    return frame


SCENES = [
    (5.0, s_intro, True),
    (5.0, s_original, True),
    (1.4, s_flip, False),
    (5.0, s_art, False),
    *[(3.4, make_closeup(*c), True) for c in CLOSEUPS],
    (5.0, s_then_now, True),
    (8.0, s_outro, True),
]
XFADE = 0.6  # cross-dissolve between scenes where flagged


# ---------------------------------------------------------------- music

def make_music(seconds, sr=44100):
    """Gentle procedural piano-ish arpeggios over a warm pad (I–V–vi–IV in C)."""
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float64)
    t_all = np.arange(n) / sr
    chords = [[60, 64, 67, 72], [55, 59, 62, 67], [57, 60, 64, 69], [53, 57, 60, 65]]
    beat = 0.45
    bar = beat * 8

    def hz(m):
        return 440 * 2 ** ((m - 69) / 12)

    def note(m, start, length, vol):
        s = int(start * sr)
        L = min(int(length * sr), n - s)
        if L <= 0:
            return
        tt = np.arange(L) / sr
        f = hz(m)
        tone = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(4 * np.pi * f * tt)
                + 0.12 * np.sin(6 * np.pi * f * tt))
        env = np.exp(-tt * 2.2) * (1 - np.exp(-tt * 200))
        out[s:s + L] += tone * env * vol

    pattern = [0, 1, 2, 3, 2, 1, 2, 3]
    i = 0
    while i * bar < seconds:
        ch = chords[i % 4]
        start = i * bar
        # pad
        s, e = int(start * sr), min(int((start + bar) * sr), n)
        tt = t_all[s:e] - start
        env = np.minimum(1, tt / 0.8) * np.minimum(1, (bar - tt) / 0.8)
        for m in ch[:3]:
            out[s:e] += 0.05 * env * np.sin(2 * np.pi * hz(m - 12) * t_all[s:e])
        # arpeggio
        for k, idx in enumerate(pattern):
            note(ch[idx] + (12 if i >= 2 and k == 7 else 0), start + k * beat, 2.5, 0.16)
        # melody on top from the 3rd bar
        if i >= 2:
            note(ch[3] + 12, start, 3.0, 0.10)
            note(ch[2] + 12, start + beat * 4, 3.0, 0.08)
        i += 1

    # simple echo for space
    d = int(0.3 * sr)
    out[d:] += 0.3 * out[:-d]
    fade = np.ones(n)
    fi, fo = int(1.5 * sr), int(3.5 * sr)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo)
    out *= fade
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.8
    pcm = (out * 32767).astype(np.int16)
    with wave.open(str(AUDIO), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- render

def main():
    # Timeline with overlaps for cross-dissolves
    timeline, start = [], 0.0
    for dur, fn, xf in SCENES:
        if timeline and xf:
            start -= XFADE
        timeline.append((start, dur, fn))
        start += dur
    total = start
    make_music(total)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(AUDIO),
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    nframes = int(total * FPS)
    for fi in range(nframes):
        t = fi / FPS
        active = [(s, d, fn) for s, d, fn in timeline if s <= t < s + d]
        s, d, fn = active[0]
        frame = fn(t - s, d)
        if len(active) > 1:
            s2, d2, fn2 = active[1]
            a = ease((t - s2) / XFADE)
            frame = Image.blend(frame, fn2(t - s2, d2), a)
        fade = ease(t / 0.8) * ease((total - t) / 1.5)
        proc.stdin.write(finish(frame, fade).tobytes())
        if fi % 150 == 0:
            print(f"frame {fi}/{nframes}", flush=True)
    proc.stdin.close()
    proc.wait()
    AUDIO.unlink(missing_ok=True)
    print(f"done: {OUT} ({total:.1f}s)")


if __name__ == "__main__":
    main()

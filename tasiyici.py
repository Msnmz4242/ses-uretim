"""Taşıyıcı cümle: zor kelimeyi 'هَذَا <kelime>' gibi bir cümlenin sonunda okutup kelimeyi kesip alır.
(Model kelimeyi tek başına okurken ج'yi 'j' yapıyor, cümle içinde 'c' okuyor.)"""
import re

import numpy as np
import torch

import fonem

TASIYICILAR = ("هَذَا", "بَيْتٌ")


def emisyonlar(fon, wav, sr):
    """[(fonem, saniye)] — CTC'nin her yeni çıktısı ve zamanı."""
    from scipy.signal import resample_poly
    a = resample_poly(np.asarray(wav, dtype=np.float32), 16000, sr).astype(np.float32)
    pad = int(0.15 * 16000)
    a = np.concatenate([np.zeros(pad, np.float32), a, np.zeros(pad, np.float32)])
    x = fon.proc(a, sampling_rate=16000, return_tensors="pt").input_values.to(fon.device)
    with torch.no_grad():
        ids = fon.model(x).logits.argmax(-1)[0].tolist()
    tok = fon.proc.tokenizer.convert_ids_to_tokens(ids)
    blank = fon.proc.tokenizer.pad_token
    n_kare = len(ids)
    kare_sn = (len(a) / 16000) / n_kare
    out, prev = [], None
    for i, t in enumerate(tok):
        if t != prev and t not in (blank, "|", " "):
            out.append((t, i * kare_sn - 0.15))
        prev = t
    return out


def ilk_sinif(kelime):
    for ch, mk in re.findall(f"([{fonem.LETTER}])([{fonem.MARK}]*)", kelime):
        c = fonem.HARF_SINIF.get(ch, "")
        if c:
            return c
    return ""


def kes(fon, wav, sr, kelime, tasiyici_payi):
    """Son kelimenin başından keser. tasiyici_payi: taşıyıcının metindeki payı (0..1), beklenen sınır."""
    hedef = ilk_sinif(kelime)
    em = emisyonlar(fon, wav, sr)
    sure = len(wav) / sr
    beklenen = tasiyici_payi * sure
    aday = []
    for t, s in em:
        b = re.sub("[ːˤ0-9.]", "", t)
        c = fonem.FONEM_SINIF.get(b, "") or {"ɕ": "ʃ", "tɕ": "ʒ", "tʃ": "ʒ", "ʁ": "ɣ"}.get(b, "")
        if c == hedef or (hedef == "ʒ" and b == "d"):
            aday.append(s)
    if not aday:
        return None
    s0 = min(aday, key=lambda s: abs(s - beklenen))
    if abs(s0 - beklenen) > 0.35 * sure:
        return None
    # s0'dan önceki 180 ms içinde enerjinin en düşük olduğu yer (ünsüzün kapanması) = kesim
    fr = int(0.01 * sr)
    e = np.array([np.sqrt(np.mean(wav[i:i + fr] ** 2)) for i in range(0, len(wav) - fr, fr)])
    a, b = max(0, int((s0 - 0.18) * 100)), min(len(e) - 1, int((s0 + 0.03) * 100))
    if b <= a:
        return None
    k = a + int(np.argmin(e[a:b + 1]))
    cut = max(0, k * fr - int(0.005 * sr))
    w = wav[cut:].copy()
    n = int(0.008 * sr)
    w[:n] *= np.linspace(0, 1, n)
    return w

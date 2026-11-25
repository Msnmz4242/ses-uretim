"""Ünlü denetçisi: sesi dilden bağımsız fonem tanımayla (wav2vec2 XLSR-53 espeak, Apache 2.0) çözer,
duyulan ünlüleri (a/i/u) metnin harekeleriyle karşılaştırır. Yalnız kontrol için; uygulamaya girmez.
"""
import difflib, re

import numpy as np
import torch

A = set("aæɑɐʌɒ")          # fetha
I = set("iɪeɛɨyø")         # kesra
U = set("uʊoɔɵ")           # damma
LETTER = "ء-ي"
MARK = "ً-ْٰ"
VOWEL_OF = {"َ": "a", "ً": "a", "ِ": "i", "ٍ": "i", "ُ": "u", "ٌ": "u", "ٰ": "a"}
WEAK_END = set("هءأؤئ")     # vakfta sonu zor duyulan ünsüzler: son kontrolü yapılmaz


def _units(w):
    return re.findall(f"([{LETTER}])([{MARK}]*)", w)


def hedef(text):
    """Metindeki ünlüler sırayla ('a','i','u') ve sonun türü: 'unlu', 'unsuz' ya da 'zayif'."""
    t = re.sub(f"[^{LETTER}{MARK} ]", " ", text.replace("‌", ""))
    words = t.split()
    out = []
    for wi, w in enumerate(words):
        us = _units(w)
        for n, (ch, mk) in enumerate(us):
            mk = mk.replace("ّ", "")
            v = VOWEL_OF.get(mk[-1:], "") if mk else ""
            if v:
                out.append(v)
            elif ch == "آ":
                out.append("a")
            elif n == 0 and ch == "ا" and wi == 0 and len(us) > 1:   # baştaki vasl elifi okunur
                nxt = us[1][0]
                third = us[2][1] if len(us) > 2 else ""
                out.append("a" if nxt == "ل" else ("u" if "ُ" in third else "i"))
    son = "unsuz"
    if words:
        us = _units(words[-1])
        if us:
            ch, mk = us[-1]
            mk2 = mk.replace("ّ", "")
            prev = us[-2][1] if len(us) > 1 else ""
            if mk2 and mk2[-1] in "َُِ":
                son = "unlu"
            elif mk2 and mk2[-1] in "ًٌٍ":
                son = "zayif"                                        # tenvin: -n ile biter
            elif not mk2 and (ch in "اىآ" or (ch == "و" and "ُ" in prev) or (ch == "ي" and "ِ" in prev)):
                son = "unlu"                                         # uzun ünlüyle biter
            elif ch in WEAK_END:
                son = "zayif"
    return out, son


def sinif(ph):
    """Fonemler → ünlü sınıfları; art arda aynı ünlü tek sayılır (aː, a a)."""
    out, prev = [], None
    for p in ph:
        base = re.sub("[ːˤ0-9]", "", p)
        c = None
        if base[:1] in A:
            c = "a"
        elif base[:1] in I:
            c = "i"
        elif base[:1] in U:
            c = "u"
        if c:
            if prev != c:
                out.append(c)
            prev = c
        elif base not in ("ə", ""):
            prev = None
    return out


def son_unlu_mu(ph):
    for p in reversed(ph):
        base = re.sub("[ːˤ0-9]", "", p)
        if base in ("ə", ""):
            continue
        return base[:1] in A | I | U
    return False


class Fonem:
    def __init__(self, device="cpu"):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        mid = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = Wav2Vec2ForCTC.from_pretrained(mid).to(device).eval()
        self.device = device

    def duy(self, wav, sr):
        from scipy.signal import resample_poly
        a = resample_poly(np.asarray(wav, dtype=np.float32), 16000, sr).astype(np.float32)
        pad = np.zeros(int(0.15 * 16000), dtype=np.float32)
        a = np.concatenate([pad, a, pad])
        x = self.proc(a, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            ids = self.model(x).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0].split()


# ünsüz sınıfları: harf → kaba sınıf (vurgulu/vurgusuz aynı sınıf; hemze güvenilmez, sayılmaz)
HARF_SINIF = {"ب": "b", "ت": "t", "ط": "t", "ث": "θ", "ج": "ʒ", "ح": "ħ", "خ": "x", "د": "d", "ض": "d", "ذ": "ð",
              "ظ": "ð", "ر": "r", "ز": "z", "س": "s", "ص": "s", "ش": "ʃ", "ع": "ʕ", "غ": "ɣ", "ف": "f", "ق": "q",
              "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "h", "و": "w", "ي": "j", "ى": ""}
FONEM_SINIF = {"b": "b", "p": "b", "t": "t", "θ": "θ", "ʒ": "ʒ", "dʒ": "ʒ", "d͡ʒ": "ʒ", "ɟ": "ʒ", "g": "ʒ", "ħ": "ħ",
               "x": "x", "χ": "x", "d": "d", "ð": "ð", "r": "r", "ɾ": "r", "z": "z", "s": "s", "ʃ": "ʃ", "ʕ": "ʕ",
               "ɣ": "ɣ", "ʁ": "ɣ", "f": "f", "q": "q", "k": "k", "l": "l", "m": "m", "n": "n", "h": "h", "w": "w",
               "j": "j", "v": "f"}


def hedef_unsuz(text):
    """Metindeki söylenen ünsüzler (uzun ünlü harfleri ve sondaki zayıf ünsüz hariç)."""
    t = re.sub(f"[^{LETTER}{MARK} ]", " ", text.replace("‌", ""))
    words = t.split()
    out = []
    for wi, w in enumerate(words):
        us = _units(w)
        for n, (ch, mk) in enumerate(us):
            prev = us[n - 1][1] if n else ""
            if ch in "اآإأءئؤى":
                continue
            if ch == "و" and not mk and "ُ" in prev:
                continue                                          # uzun û
            if ch == "ي" and not mk and "ِ" in prev:
                continue                                          # uzun î
            if ch == "ل" and n == 1 and us[0][0] == "ا" and not mk:
                continue                                          # güneş harfine kaynaşan lam
            if wi == len(words) - 1 and n == len(us) - 1 and ch in "هة" and "َ" not in mk + "":
                continue                                          # vakfta sondaki h zor duyulur
            c = HARF_SINIF.get(ch, "")
            if c and not (out and out[-1] == c and "ّ" in mk):
                out.append(c)
    return [c for i, c in enumerate(out) if i == 0 or c != out[i - 1]]


def duyulan_unsuz(ph):
    out = []
    for p in ph:
        base = re.sub("[ːˤ0-9]", "", p)
        c = FONEM_SINIF.get(base) or FONEM_SINIF.get(base[:1] if base[:1] not in A | I | U else "", "")
        if c and (not out or out[-1] != c):
            out.append(c)
    return out


def olc(text, ph):
    """usim: ünlü dizisi benzerliği (0..1); csim: ünsüz benzerliği; fazla_son: sükûnlu sona ünlü eklenmiş;
    eksik_son: sondaki ünlü yok."""
    tv, son = hedef(text)
    hv = sinif(ph)
    sim = difflib.SequenceMatcher(None, tv, hv).ratio() if (tv or hv) else 1.0
    tc, hc = hedef_unsuz(text), duyulan_unsuz(ph)
    csim = difflib.SequenceMatcher(None, tc, hc).ratio() if (tc or hc) else 1.0
    end_v = son_unlu_mu(ph)
    return {"usim": round(sim, 3), "utam": tv == hv, "csim": round(csim, 3), "fazla_son": son == "unsuz" and end_v,
            "eksik_son": son == "unlu" and not end_v, "hedef": "".join(tv), "duyulan": "".join(hv),
            "hedef_c": "".join(tc), "duyulan_c": "".join(hc), "fonem": " ".join(ph)}


def duz_mu(wav):
    """Konuşma değil de sabit uğultu/gürültü mü (enerji hiç değişmiyor)."""
    w = np.asarray(wav, dtype=np.float32)
    fr = w[: len(w) // 480 * 480].reshape(-1, 480)
    if len(fr) < 4:
        return True
    e = np.sqrt((fr ** 2).mean(1))[1:]
    return float(e.std() / (e.mean() + 1e-9)) < 0.2

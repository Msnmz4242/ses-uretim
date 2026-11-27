"""Sıkı söyleniş denetimi (kısa sesler: kelime, hece, harf adı, seçenek).

İki bağımsız kulak:
  1) harekeli Arapça tanıma (uret.Judge) — ses iki kez art arda verilir, kısa kelimede çok daha iyi duyuyor;
     yalnız ÇELİŞKİ sayılır: harekesini yazmadığı harf bilgi vermez, yazdığı hareke hedefle çelişiyorsa hata.
  2) fonem tanıma (fonem.Fonem) — ج/ث/ذ/ض harflerinin standart (fasih) okunuşu ve sona eklenen ünlü.
"""
import difflib
import re

import numpy as np

import fonem

HAREKE = {"َ": "a", "ً": "a", "ِ": "i", "ٍ": "i", "ُ": "u", "ٌ": "u", "ْ": "0"}
LETTER = "ء-ي"
MARK = "ً-ْٰ"


def _harf(ch):
    if ch in "أإآءؤئ":
        return "ء"
    return {"ة": "ه", "ى": "ا"}.get(ch, ch)


def iskelet(s):
    """[(harf, ünlü)] — ünlü: a/i/u, '0' (sükûn) ya da '' (yazılmamış)."""
    out = []
    for ch, mk in re.findall(f"([{LETTER}])([{MARK}]*)", s.replace("‌", "")):
        m = mk.replace("ّ", "")
        out.append((_harf(ch), HAREKE.get(m[-1:], "") if m else ""))
    return out


def celiski(hedef, duyulan):
    """Harekeli tanımanın hedefle çeliştiği yer sayısı (son harf hariç: tanıma oraya i'rab eklemeye meyilli)."""
    t, h = iskelet(hedef), iskelet(duyulan)
    if not t or not h:
        return 0
    sm = difflib.SequenceMatcher(None, [x[0] for x in t], [x[0] for x in h], autojunk=False)
    n = 0
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            i, j = a + k, b + k
            if i == len(t) - 1:
                continue
            tv, hv = t[i][1], h[j][1]
            uzun = t[i][0] in "اوي" and tv == ""          # uzun ünlü harfi / harekesiz harf: bilgi yok
            if uzun or not hv:
                continue
            if tv in ("a", "i", "u") and hv in ("a", "i", "u") and tv != hv:
                n += 1                                     # yanlış ünlü
            elif tv in ("0", "") and hv in ("a", "i", "u"):
                n += 1                                     # araya ünlü girmiş (مَوْز → مَوَز)
            elif tv in ("a", "i", "u") and hv == "0":
                n += 1                                     # ünlü yutulmuş
    return n


def harf_ihlali(hedef, ph):
    """Fasih okunuştan sapma: ج 'c' (dʒ), ث 'θ', ذ 'ð', ض 'd' olmalı."""
    p = {re.sub("[0-9ːˤ]", "", x) for x in ph}
    n = 0
    if "ج" in hedef and not p & {"dʒ", "d͡ʒ", "ɟ"} and p & {"ʒ", "z", "g", "ɡ", "j"}:
        n += 1
    if "ث" in hedef and "θ" not in p and p & {"s", "t", "f"}:
        n += 1
    if "ذ" in hedef and "ð" not in p and p & {"z", "d", "v"}:
        n += 1
    if "ض" in hedef and "ð" in p and "ذ" not in hedef and "ظ" not in hedef and "d" not in p:
        n += 1
    return n


def iki_kez(wav, sr):
    pad = np.zeros(int(0.6 * sr), np.float32)
    w = np.asarray(wav, dtype=np.float32)
    return np.concatenate([pad, w, pad, w, pad])


def puanla(hedef, wav, sr, judge, fon, kind="kelime"):
    """(puan, geçti_mi, ayrıntı)"""
    import uret
    if not uret.kisa_mi(hedef, kind) and kind == "cumle":
        ph = fon.duy(wav, sr)
        heard = judge.hear(wav, sr)
        s, ok, f = uret.puan(hedef, kind, heard, ph, wav, sr)
        hi = 1 if ("ج" in hedef and "ʒ" in {re.sub("[0-9ːˤ]", "", x) for x in ph}) else 0
        hi += harf_ihlali(hedef.replace("ج", ""), ph)
        return round(s - 0.3 * hi, 3), bool(ok and hi == 0), {"h2": heard, "c": 0, "hi": hi, "f": f["fonem"],
                                                               "u": f["usim"], "fs": f["fazla_son"], "es": f["eksik_son"]}
    ph = fon.duy(wav, sr)
    f = fonem.olc(hedef, ph)
    duyulan = judge.hear(iki_kez(wav, sr), sr)
    c = celiski(hedef, duyulan)
    hi = harf_ihlali(hedef, ph)
    bos = (not ph) or fonem.duz_mu(wav)
    gecti = not bos and c == 0 and hi == 0 and not f["fazla_son"] and f["usim"] >= 0.6
    puan = 0.0 if bos else (1.0 - 0.3 * c - 0.3 * hi - 0.3 * f["fazla_son"] - 0.15 * f["eksik_son"]
                            + 0.2 * f["usim"])
    return round(puan, 3), gecti, {"h2": duyulan, "c": c, "hi": hi, "f": f["fonem"], "u": f["usim"],
                                   "fs": f["fazla_son"], "es": f["eksik_son"]}

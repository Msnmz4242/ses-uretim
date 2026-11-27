"""Sıkı söyleniş denetimi (kısa sesler: kelime, hece, harf adı, seçenek).

İki bağımsız kulak:
  1) harekeli Arapça tanıma (uret.Judge) — ses iki kez art arda verilir, kısa kelimede çok daha iyi duyuyor;
     yalnız ÇELİŞKİ sayılır: harekesini yazmadığı harf bilgi vermez, yazdığı hareke hedefle çelişiyorsa hata.
  2) fonem tanıma (fonem.Fonem) — ج/ث/ذ/ض harflerinin standart (fasih) okunuşu, sona eklenen ünlü ve
     kelime içinde sükûnlu iki ünsüzün arasına giren ünlü (صَرْف → saraf).
  Ünsüzler: her ünsüzü en az bir kulak duymalı (طَاءْ 'fa', ضَ 'ba', ظَ 'va' gibi yanlış harfler).
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


GIRTLAK = set("عءأإؤئهة")
UNLU = fonem.A | fonem.I | fonem.U              # "ə" sayılmaz: sükûnda hafif çözülme (kalkale gibi) hata değil


def _taban(p):
    return re.sub("[ːˤ0-9]", "", p)


def araya_unlu(hedef, ph):
    """Kelime içinde sükûnlu ünsüzle sonraki ünsüz arasına giren ünlü sayısı (صَرْف → saraf, مَوْز → mawaz)."""
    t = re.sub(f"[^{fonem.LETTER}{fonem.MARK} ]", " ", hedef.replace("‌", ""))
    tu = []                                         # (sınıf, sükûn mu, harf, kelime no)
    for wi, w in enumerate(t.split()):
        us = fonem._units(w)
        for n, (ch, mk) in enumerate(us):
            prev = us[n - 1][1] if n else ""
            atla = (ch in "اآإأءئؤى" or (ch == "و" and not mk and "ُ" in prev) or (ch == "ي" and not mk and "ِ" in prev)
                    or (ch == "ل" and n == 1 and us[0][0] == "ا" and not mk))
            c = "" if atla else fonem.HARF_SINIF.get(ch, "")
            if not c:
                if tu and tu[-1][3] == wi:
                    tu[-1] = (tu[-1][0], False, tu[-1][2], wi)      # arada atlanan harf (hemze vb.): komşu değiller
                continue
            tu.append((c, "ْ" in mk, ch, wi))
    hs = []                                         # (sınıf, fonem sırası)
    son_unlu = -1
    for i, p in enumerate(ph):
        b = _taban(p)
        if b[:1] in UNLU or b == "ə":
            son_unlu = i
            continue
        c = fonem.FONEM_SINIF.get(b, "")
        if not c:
            continue
        if hs and hs[-1][0] == c and son_unlu < hs[-1][1]:
            continue                                # aynı ünsüz art arda (şedde/ikizleşme)
        hs.append((c, i))
    sm = difflib.SequenceMatcher(None, [x[0] for x in tu], [x[0] for x in hs], autojunk=False)
    eş = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            eş[a + k] = b + k
    n = 0
    for i in range(len(tu) - 1):
        c1, suk, h1, w1 = tu[i]
        h2, w2 = tu[i + 1][2], tu[i + 1][3]
        if not suk or w1 != w2 or i not in eş or (i + 1) not in eş or eş[i + 1] != eş[i] + 1:
            continue
        if h1 in GIRTLAK or h2 in GIRTLAK:
            continue
        arada = [_taban(p) for p in ph[hs[eş[i]][1] + 1: hs[eş[i + 1]][1]]]
        arada = [b for b in arada if b[:1] in UNLU]
        if h1 == "و":
            arada = [b for b in arada if b[:1] not in fonem.U]
        if h1 == "ي":
            arada = [b for b in arada if b[:1] not in fonem.I]
        if arada:
            n += 1
    return n


ZAYIF = set("wjh")                      # و ي ه: ünlü gibi duyulabilir / sonda düşer — sayılmaz
EK_FONEM = {"ɕ": "ʃ", "ʂ": "ʃ", "tɕ": "ʒ", "tʃ": "ʒ", "ʁ": "ɣ", "χ": "x"}


def iskelet_unsuz(s):
    out = []
    for ch, mk in re.findall(f"([{fonem.LETTER}])([{fonem.MARK}]*)", (s or "").replace("‌", "")):
        if ch in "اآإأءئؤىة":
            continue
        c = fonem.HARF_SINIF.get(ch, "")
        if c and c not in ZAYIF and not (out and out[-1] == c):
            out.append(c)
    return out


def fonem_unsuz(ph):
    out = []
    for p in ph:
        b = re.sub("[ːˤ0-9.]", "", p)
        c = EK_FONEM.get(b) or fonem.FONEM_SINIF.get(b, "")
        if c and c not in ZAYIF and not (out and out[-1] == c):
            out.append(c)
    return out


def _eslesen(t, h):
    sm = difflib.SequenceMatcher(None, t, h, autojunk=False)
    return {a + k for a, b, size in sm.get_matching_blocks() for k in range(size)}


def unsuz_onay(hedef, duyulan_metin, ph):
    """Hedefteki ünsüzlerin kaçta kaçını en az bir kulak (harekeli tanıma ya da fonem) duydu (0..1)."""
    t = iskelet_unsuz(hedef)
    if not t:
        return 1.0
    ok = _eslesen(t, iskelet_unsuz(duyulan_metin)) | _eslesen(t, fonem_unsuz(ph))
    return round(len(ok) / len(t), 3)


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
        return round(s - 0.3 * hi, 3), bool(ok and hi == 0), {"h2": heard, "c": 0, "hi": hi, "ae": 0, "f": f["fonem"],
                                                               "u": f["usim"], "fs": f["fazla_son"], "es": f["eksik_son"]}
    ph = fon.duy(wav, sr)
    f = fonem.olc(hedef, ph)
    duyulan = judge.hear(iki_kez(wav, sr), sr)
    c = celiski(hedef, duyulan)
    hi = harf_ihlali(hedef, ph)
    ae = araya_unlu(hedef, ph)
    uo = unsuz_onay(hedef, duyulan, ph)
    bos = (not ph) or fonem.duz_mu(wav)
    gecti = (not bos and c == 0 and hi == 0 and ae == 0 and uo >= 0.6 and not f["fazla_son"]
             and f["usim"] >= 0.6)
    puan = 0.0 if bos else (1.0 - 0.3 * c - 0.3 * hi - 0.3 * ae - 0.4 * (1 - uo) - 0.3 * f["fazla_son"]
                            - 0.15 * f["eksik_son"] + 0.2 * f["usim"])
    return round(puan, 3), gecti, {"h2": duyulan, "c": c, "hi": hi, "ae": ae, "uo": uo, "f": f["fonem"],
                                   "u": f["usim"], "fs": f["fazla_son"], "es": f["eksik_son"]}

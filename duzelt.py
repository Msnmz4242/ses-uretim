"""Düzeltme turu: mevcut kısa sesleri sıkı denetimden (denetim.py) geçirir; geçemeyenleri yeni denemelerle
yeniden üretir ve en iyisini seçer. Mevcut ses denetimden geçerse dokunulmaz.

Kullanım: python duzelt.py --metinler m.json --mevcut <eski ogg klasörü> --ref ref.wav --cikti ses_out --parca i/n --sure s
Çıktı: yalnız değişen sesler (<anahtar>.ogg) + index_<i>.json (değişenler) + denetim_<i>.json (hepsinin sonucu)
"""
import argparse, json, os, subprocess, time

import numpy as np
import soundfile as sf
import torch

import denetim
import fonem
import tasiyici
import uret

TOHUMLAR = tuple(int(x) for x in os.environ.get("SES_TOHUMLAR", "131,137,139,149,151,157,163,167").split(","))
PLAN = os.environ.get("SES_PLAN", "1")           # 2: ikinci tur (ج için 'دْج' yazımı, iki hız, daha çok deneme)
ISARET = "ًٌٍَُِّْٰ"
KISA_UNLU = "َُِ"


def hileli(say):
    """ج'yi modele 'دْج' diye yazar (c sesi = d + j): yalnız önünde aynı kelimede ünlüyle biten bir harf varsa
    (تَجْرِبَة → تَدْجْرِبَة, مُتَزَوِّج → مُتَزَوِّدْج). Kelime başında ya da sükûnlu harften sonra araya ünlü girdiği
    için dokunulmaz. Denetim her zaman asıl metne göre yapılır."""
    out = []
    for i, ch in enumerate(say):
        if ch == "ج":
            j, kendi = i + 1, ""
            while j < len(say) and say[j] in ISARET:
                kendi += say[j]
                j += 1
            k, onceki = i - 1, ""
            while k >= 0 and say[k] in ISARET:
                onceki = say[k] + onceki
                k -= 1
            harf = say[k] if k >= 0 else " "
            bas = k <= 0 or not ("ء" <= say[k - 1] <= "ي" or say[k - 1] in ISARET)     # kelimenin ilk harfi mi
            unlu = any(m in KISA_UNLU for m in onceki) or (harf in "اىوي" and "ْ" not in onceki and not bas)
            if "ّ" not in kendi and "ء" <= harf <= "ي" and unlu:
                out.append("دْج")
                continue
        out.append(ch)
    return "".join(out)


def denemeler(say, kind):
    """[(tohum, örnek no, hile mi, tempo)]"""
    if PLAN == "3":                                    # taşıyıcı cümle: örnekler ve hile sırayla
        var = hileli(say) != say
        return [(s, 2 if n % 2 == 0 else 1, var and n % 4 in (0, 3), 1.0) for n, s in enumerate(TOHUMLAR)]
    if PLAN != "2":
        cift = any(h in say for h in "جذ")
        t = TOHUMLAR if kind != "cumle" else TOHUMLAR[:4]
        return [(s, 2 if (cift and n % 2 == 1) else 1, False, None) for n, s in enumerate(t)]
    var = hileli(say) != say
    t = TOHUMLAR if kind != "cumle" else TOHUMLAR[:6]
    out = []
    for n, s in enumerate(t):
        if "ج" in say:
            ref = 1 if n % 3 == 2 else 2                  # çoğunlukla ikinci örnek (ج'yi c okuyor)
            hile = var and n % 2 == 0
        elif "ذ" in say:
            ref, hile = (2 if n % 2 == 0 else 1), False
        else:
            ref, hile = (1 if n % 2 == 0 else 2), False   # ث: ilk örnekte θ var (الثِّمَارُ)
        tempo = 1.0 if (kind == "cumle" or (n // 2) % 2 == 0) else 0.85   # cümlede hız aynı kalır
        out.append((s, ref, hile, tempo))
    return out
# ikinci örnek: aynı konuşmacı (Arabic Speech Corpus, Nawar Halabi), ج ve ذ'yi fasih okuduğu bir cümle
REF2_TEXT = "أَتَاحَتْ لِلْبَائِعِ الْمُتَجَوِّلِ أَنْ يَكُونَ جَاذِبًا لِلْمُوَاطِنِ الْأَقَلِّ دَخْلًا"


def oku(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "24000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32), 24000


def yaz(wav, sr, path, tmp):
    thr = 0.02 * max(1e-6, float(np.abs(wav).max()))
    nz = np.where(np.abs(wav) > thr)[0]
    if len(nz):
        wav = wav[max(0, nz[0] - int(0.04 * sr)): nz[-1] + int(0.12 * sr)]
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
    wav = wav * min(0.09 / rms, 0.9 / (float(np.abs(wav).max()) + 1e-9))
    sf.write(tmp, wav, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-c:a", "libopus", "-b:a", "40k",
                    "-application", "audio", path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metinler", default="metinler.json")
    ap.add_argument("--mevcut", default="mevcut")
    ap.add_argument("--ref", default="ref.wav")
    ap.add_argument("--ref2", default="ref2.wav")
    ap.add_argument("--cikti", default="ses_out")
    ap.add_argument("--parca", default="0/1")
    ap.add_argument("--sure", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = uret.Judge(device)
    fon = fonem.Fonem(device)
    tts = None
    items = json.load(open(args.metinler, encoding="utf-8"))
    i, n = map(int, args.parca.split("/"))
    items = [x for j, x in enumerate(items) if j % n == i]
    os.makedirs(args.cikti, exist_ok=True)
    idx_path = os.path.join(args.cikti, f"index_{i}.json")
    den_path = os.path.join(args.cikti, f"denetim_{i}.json")
    index, den = {}, {}
    tmp = os.path.join(args.cikti, "_tmp.wav")
    t0 = time.time()
    for c, it in enumerate(items, 1):
        if args.sure and time.time() - t0 > args.sure:
            uret.report("süre doldu, burada durdu")
            break
        text, kind = it["t"], it["k"]
        k = uret.key(text)
        say = uret.okunus(text, kind)
        try:
            best = None
            eski = os.path.join(args.mevcut, k + ".ogg")
            if os.path.exists(eski) and not it.get("z"):           # z: okunuşu değişti, eskisi geçersiz
                w, sr = oku(eski)
                p, ok, d = denetim.puanla(say, w, sr, judge, fon, kind)
                best = (p, ok, "eski", None, sr, d, 0)
            if best is None or not best[1]:
                if tts is None:
                    import silma_tts.infer.utils_infer as U
                    U.normalize_last_word = lambda t: t.strip()
                    from silma_tts.api import SilmaTTS
                    tts = SilmaTTS(device=device)
                for n_try, (seed, ref, hile, tempo) in enumerate(denemeler(say, kind), 1):
                    if ref == 2 and not os.path.exists(args.ref2):
                        ref = 1
                    gen = hileli(say) if hile else say
                    if PLAN == "3":
                        gen = f"{tasiyici.TASIYICILAR[0]} {gen}"
                    wav, sr, _ = tts.infer(ref_file=args.ref2 if ref == 2 else args.ref,
                                           ref_text=REF2_TEXT if ref == 2 else uret.REF_TEXT, gen_text=gen, file_wave=tmp,
                                           seed=seed, nfe_step=16, remove_silence=False,
                                           show_info=lambda *a, **kw: None, speed=uret.hiz(gen, tempo),
                                           force_tashkeel=False)
                    wav = np.asarray(wav, dtype=np.float32)
                    if PLAN == "3":                            # kelimeyi taşıyıcı cümleden kes
                        pay = (len(tasiyici.TASIYICILAR[0]) + 1) / (len(gen) + 2)
                        wav = tasiyici.kes(fon, wav, sr, say, pay)
                        if wav is None:
                            continue
                    p, ok, d = denetim.puanla(say, wav, sr, judge, fon, kind)
                    d.update(ref=ref, hile=int(hile), tempo=tempo or uret.TEMPO)
                    if best is None or p > best[0] or (ok and not best[1]):
                        best = (p, ok, "yeni", wav, sr, d, n_try)
                    if ok:
                        break
            p, ok, src, wav, sr, d, n_try = best
            den[k] = {"t": text, "v": say, "k": kind, "kaynak": src, "p": p, "ok": ok, "n": n_try, **d}
            if src == "yeni":
                yaz(wav, sr, os.path.join(args.cikti, k + ".ogg"), tmp)
                index[k] = {"t": text, "v": say, "k": kind, "ok": bool(ok), "s": p, "n": n_try, "h": d["h2"],
                            "f": d["f"], "u": d["u"], "c": d["c"], "hi": d["hi"], "ae": d.get("ae", 0), "fs": d["fs"],
                            "es": d["es"], "ref": d.get("ref"), "hile": d.get("hile", 0), "tempo": d.get("tempo")}
        except Exception:
            import traceback
            traceback.print_exc()
            uret.report(f"{c}/{len(items)} hata, atlandı")
        if c % 10 == 0 or c == len(items):
            json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False)
            json.dump(den, open(den_path, "w", encoding="utf-8"), ensure_ascii=False)
            uret.report(f"{c}/{len(items)}  {(time.time() - t0) / 60:.1f} dk  değişen: {len(index)}")
    json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(den, open(den_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

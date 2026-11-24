"""Lisan ses üretimi: metinler.json → kontrol edilmiş Opus ses dosyaları.

Ses: SILMA TTS (Apache 2.0) + örnek ses Arabic Speech Corpus, Nawar Halabi (CC BY 4.0).
Her metin için birkaç deneme üretilir; harekeli Arapça ses tanıma ile metne en çok
uyan seçilir. Yarıda kalırsa tekrar çalıştırmak kaldığı yerden devam eder.

Kullanım:
  python uret.py --metinler metinler.json --ref ref.wav --cikti ses_out [--parca 0/1] [--sinir 20]
"""
import argparse, hashlib, json, os, re, subprocess, sys, time, difflib

import numpy as np
import soundfile as sf
import torch

REF_TEXT = "وَيُمْكِنُ أَنْ تُحْصَدَ الثِّمَارُ عَلَى مَدًى فَتْرَةٍ زَمَنِيَّةٍ طَوِيلَةٍ"
HARAKAT = "ًٌٍَُِّْ"
SEEDS = (7, 11, 23, 37)


SUKUN = "\u0652"
SHORT = "\u064C\u064D\u064E\u064F\u0650"          # ٌ ٍ َ ُ ِ


def vakf(text):
    """Son kelimeyi duraklama (vakf) kuralıyla yazar; cümle içindeki kelimeler olduğu gibi kalır.
    ـةٌ/ـةً/ـةِ → ـهْ · ـًا / ـًى / ـءً → ـَا · diğer son kısa hareke/tenvin → sükûn."""
    m = re.match(r"^(.*?)([\s،؟!.,:؛]*)$", text, re.S)
    body, tail = m.group(1), m.group(2)
    if not body:
        return text
    i = len(body)
    while i > 0 and body[i - 1] in HARAKAT + "\u0670":
        i -= 1
    j = i                                             # son harfin başladığı yer (harekeler hariç)
    letter, marks = body[i - 1: i], body[i:]
    head = body[: i - 1]
    if not letter or not re.match("[\u0621-\u064A]", letter):
        return text
    if letter == "ة":
        return head + "هْ" + tail
    pm = re.search("[" + HARAKAT + "]*$", head).group(0)              # bir önceki harfin harekeleri
    if letter in "اى" and "\u064B" in pm:                            # ـًا / ـًّا / ـًى
        return head[: len(head) - len(pm)] + pm.replace("\u064B", "\u064E") + letter + tail
    if "\u064B" in marks:                                           # ـاً / ـءً
        if letter in "اى":
            return head + "\u064E" * 0 + letter + tail
        return head + letter + "\u064E" + "ا" + tail
    keep = marks.replace("\u0651", "")
    if keep and keep[-1] in SHORT:
        return head + letter + ("\u0651" if "\u0651" in marks else "") + SUKUN + tail
    return text


_PROGRESS_FD = int(os.environ.get("ILERLEME_FD", "0") or 0)


def report(msg):
    """Yalnızca sayılar içeren ilerleme satırı (herkese açık kayıtlarda metin görünmesin)."""
    print(msg, flush=True)
    if _PROGRESS_FD:
        os.write(_PROGRESS_FD, (msg + "\n").encode())


def key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def norm(s):
    s = re.sub("[ً-ْٰـ]", "", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub("[^ء-ي]", "", s)


def cons(s):
    """Ünsüz iskeleti: tanıma programı harekeleri harf gibi yazabildiği için uzun ünlüler de çıkarılır."""
    return re.sub("[اويه]", "", norm(s))


def vowels(s):
    """Her harfin harekesi sırayla (a/i/u/0/an/un/in); harekesiz harfler atlanır."""
    out = []
    for ch, mk in re.findall("([\u0621-\u064A])([" + HARAKAT + "]*)", s):
        mk = mk.replace("\u0651", "")
        if mk:
            out.append({"\u064B": "an", "\u064C": "un", "\u064D": "in", "\u064E": "a", "\u064F": "u",
                        "\u0650": "i", "\u0652": "0"}.get(mk[-1], ""))
    return out


def last_mark(s):
    """Son kelimenin son harekesi ('' = harekesiz / sükûn)."""
    w = re.sub("[^ء-ي" + HARAKAT + " ]", "", s).strip().split(" ")[-1] if s.strip() else ""
    m = re.findall("[" + HARAKAT + "]+$", w)
    m = m[0].replace("ّ", "") if m else ""
    return {"ً": "an", "ٌ": "un", "ٍ": "in", "َ": "a", "ُ": "u", "ِ": "i"}.get(m[-1:] if m else "", "")


class Judge:
    """Harekeli Arapça ses tanıma (yalnız kontrol için kullanılır, uygulamaya girmez)."""

    def __init__(self, device):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        mid = "NightPrince/stt-arabic-whisper-finetuned-diactires"
        self.proc = WhisperProcessor.from_pretrained(mid)
        self.model = WhisperForConditionalGeneration.from_pretrained(mid).to(device).eval()
        self.device = device

    def hear(self, wav, sr):
        from scipy.signal import resample_poly
        a = resample_poly(wav, 16000, sr).astype(np.float32)
        f = self.proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(f, max_new_tokens=120, language="ar", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def score(text, kind, heard, dur):
    """0..1 arası puan ve geçti mi."""
    short = kind in ("harf", "hece") or len(norm(text)) <= 2
    exp = 0.45 + 0.085 * len(norm(text))
    if dur < 0.2 or dur > exp * 2.2 + 0.6:
        return 0.0, False                                   # boş ya da fazladan ses eklenmiş
    if short:
        return 0.8, True                                    # hece / harf: tanıma güvenilir değil, süreye bak
    sim = difflib.SequenceMatcher(None, cons(text), cons(heard)).ratio()
    want, got = last_mark(text), last_mark(heard)
    end_ok = not want or not got or want == got or (want in ("an", "un", "in") and got == want[0])
    # tanıma çıktısı harekeliyse kelime içi harekeleri de karşılaştır (مَشَى ≠ مِشْي)
    hv, tv = vowels(heard), vowels(text)
    marked = len(hv) >= 0.4 * max(1, len(norm(heard)))
    vsim = difflib.SequenceMatcher(None, tv, hv).ratio() if marked and tv else 1.0
    s = 0.75 * sim + 0.25 * vsim - (0 if end_ok else 0.15)
    return s, sim >= 0.9 and end_ok and vsim >= 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metinler", default="metinler.json")
    ap.add_argument("--ref", default="ref.wav")
    ap.add_argument("--cikti", default="ses_out")
    ap.add_argument("--parca", default="0/1", help="i/n: metinlerin n parçadan i'incisi")
    ap.add_argument("--sinir", type=int, default=0, help="en fazla bu kadar metin (deneme için)")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--sure", type=int, default=0, help="en fazla bu kadar saniye çalış (0 = sınırsız)")
    ap.add_argument("--usta", type=int, default=1, help="yüksek kaliteli asıl kopyaları da sakla (1/0)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report(f"cihaz: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''} · çekirdek: {os.cpu_count()}")
    import silma_tts.infer.utils_infer as U
    U.normalize_last_word = lambda text: text.strip()       # modelin kendi vakf'ı kapalı: vakf'ı vakf() yapıyor
    from silma_tts.api import SilmaTTS
    tts = SilmaTTS(device=device)
    judge = Judge(device)

    items = json.load(open(args.metinler, encoding="utf-8"))
    i, n = map(int, args.parca.split("/"))
    items = [x for j, x in enumerate(items) if j % n == i]
    if args.sinir:
        items = items[: args.sinir]
    os.makedirs(args.cikti, exist_ok=True)
    idx_path = os.path.join(args.cikti, f"index_{i}.json")
    index = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {}
    tmp = os.path.join(args.cikti, "_tmp.wav")
    t_start = time.time()
    for c, it in enumerate(items, 1):
        if args.sure and time.time() - t_start > args.sure:
            report("süre doldu, burada durdu")
            break
        text, kind = it["t"], it["k"]
        k = key(text)                                      # anahtar: uygulamanın okuttuğu metin
        say = text if kind in ("harf", "hece") else vakf(text)   # okunan: son kelime vakf ile
        if k in index and os.path.exists(os.path.join(args.cikti, k + ".ogg")):
            continue
        try:
            best = None
            for n_try, seed in enumerate(SEEDS, 1):
                wav, sr, _ = tts.infer(ref_file=args.ref, ref_text=REF_TEXT, gen_text=say, file_wave=tmp, seed=seed,
                                       nfe_step=args.nfe, remove_silence=False, show_info=lambda *a, **k: None)
                wav = np.asarray(wav, dtype=np.float32)
                heard = judge.hear(wav, sr) if kind not in ("harf", "hece") else ""
                s, ok = score(say, kind, heard, len(wav) / sr)
                if best is None or s > best[0]:
                    best = (s, ok, wav, sr, heard, n_try)
                if ok:
                    break
            s, ok, wav, sr, heard, n_try = best
            # baştaki/sondaki sessizliği kırp
            thr = 0.02 * max(1e-6, float(np.abs(wav).max()))
            nz = np.where(np.abs(wav) > thr)[0]
            if len(nz):
                wav = wav[max(0, nz[0] - int(0.04 * sr)): nz[-1] + int(0.12 * sr)]
            # yalnızca ses seviyesini eşitle (tepe 0.9'u geçmeyecek şekilde); ses rengine dokunma
            rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
            peak = float(np.abs(wav).max() + 1e-9)
            wav = wav * min(0.09 / rms, 0.9 / peak)
            sf.write(tmp, wav, sr)
            if args.usta:                               # yüksek kaliteli asıl kopya: sonradan yeniden sıkıştırmak için
                os.makedirs(os.path.join(args.cikti, "usta"), exist_ok=True)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-c:a", "libopus", "-b:a", "96k",
                                "-application", "audio", os.path.join(args.cikti, "usta", k + ".ogg")], check=True)
            # "audio" modu: konuşma (voip) modu sesin rengini değiştiriyordu
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-c:a", "libopus", "-b:a", "40k",
                            "-application", "audio", os.path.join(args.cikti, k + ".ogg")], check=True)
            index[k] = {"t": text, "v": say, "k": kind, "ok": bool(ok), "s": round(float(s), 3), "n": n_try, "h": heard}
        except Exception:
            import traceback
            traceback.print_exc()                       # günlüğe gider (şifreli)
            report(f"{c}/{len(items)} hata, atlandı")
        if c % 10 == 0 or c == len(items):
            json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False)
            el = time.time() - t_start
            report(f"{c}/{len(items)}  {el / 60:.1f} dk  zayıf: {sum(1 for v in index.values() if not v['ok'])}")
    json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main()

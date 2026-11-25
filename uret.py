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
SEEDS = tuple(int(x) for x in os.environ.get("SES_TOHUMLAR", "7,11,23,37").split(","))


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


MARKS = HARAKAT + "ٰ"
PUNCT = "،؟!.,:؛?;"
# seçenek olarak tek başına gösterilen harf → harfin adı (uygulamadaki harf kartıyla aynı)
HARF_ADI = {
    "ا": "أَلِف", "ب": "بَاء", "ت": "تَاء", "ث": "ثَاء", "ج": "جِيم", "ح": "حَاء", "خ": "خَاء",
    "د": "دَال", "ذ": "ذَال", "ر": "رَاء", "ز": "زَاي", "س": "سِين", "ش": "شِين", "ص": "صَاد",
    "ض": "ضَاد", "ط": "طَاء", "ظ": "ظَاء", "ع": "عَيْن", "غ": "غَيْن", "ف": "فَاء", "ق": "قَاف",
    "ك": "كَاف", "ل": "لَام", "م": "مِيم", "ن": "نُون", "ه": "هَاء", "و": "وَاو", "ي": "يَاء",
    "ة": "تَاء مَرْبُوطَة", "ء": "هَمْزَة", "لا": "لَام أَلِف",
}
SES_HARF = {"أ": "أَ", "إ": "إِ"}                  # hemzeli elif: sesi okunur
GOSTERIM = {"بْ": "أَبْ", "بّ": "أَبَّ"}           # sükûn / şedde tek başına okunamaz: ab, abba


def son_sukun(w):
    """Sonu harekesiz kalan ünsüze açıkça sükûn koyar; yoksa model kendi ünlüsünü ekliyor."""
    if len(w) > 1 and w[-1] == "ّ" and re.match("[ء-ي]", w[-2]):
        return w + SUKUN                                # فَنّ → فَنّْ (vakf'taki şeddeli sonla aynı)
    if not w or w[-1] in MARKS or not re.match("[ء-ي]", w[-1]):
        return w
    last = w[-1]
    prev = re.search("[" + MARKS + "]*$", w[:-1]).group(0)
    if last in "اىآ" or (last == "و" and "ُ" in prev) or (last == "ي" and "ِ" in prev):
        return w                                        # uzun ünlü
    if last == "ة":
        return w[:-1] + "ه" + SUKUN
    return w + SUKUN


def pausa(text):
    """Duraklamadan önceki her kelimeye vakf, harekesiz kalan son ünsüzlere sükûn."""
    toks = text.split(" ")
    out = []
    for n, tok in enumerate(toks):
        m = re.match(r"^(.*?)([" + PUNCT + r"]*)$", tok, re.S)
        w, p = m.group(1), m.group(2)
        if w and (p or n == len(toks) - 1):
            w = vakf(w)
        out.append(son_sukun(w) + p)
    return " ".join(out)


def okunus(text, kind):
    """Uygulamanın gösterdiği metin → modele okutulan metin."""
    t = text.replace("‌", "").strip()                 # ZWNJ: yalnız anahtarı ayırmak için
    if t in GOSTERIM:
        return GOSTERIM[t]
    if kind == "secenek" and t in SES_HARF:
        return SES_HARF[t]
    if kind == "secenek" and t in HARF_ADI:
        return pausa(HARF_ADI[t])
    if kind in ("harf", "hece"):
        if len(norm(t)) <= 1:
            return t                                    # بَ بِ بُ بٌ: hareke gösterimi aynen
        if kind == "harf":
            return pausa(t)                             # harf adı: بَاءْ، تَاءْ مَرْبُوطَهْ
        return " ".join(son_sukun(w) for w in t.split(" "))   # hece: son ünlü okunur (جَلَسَ)
    return pausa(t)


TEMPO = float(os.environ.get("SES_HIZ", "1.0"))        # 1.0 = örnek sesteki konuşma hızı


def hiz(say, tempo=None):
    """SILMA'nın kendiliğinden uyguladığı hız çarpanlarını geri alır; konuşma hızı = tempo.
    (harekeli metin ×1.3 hızlandırılıyor, 25 karakterden kısa metin ÷1.3 yavaşlatılıyor)"""
    tempo = TEMPO if tempo is None else tempo
    g = say.strip()
    marks = len(re.findall("[ً-ْ]", g))
    factor = 1.3 if (marks >= 3 and marks / len(g) > 0.2) else 1.0
    b = g if g[-1] in ".,،؟؛:!?;" else g + ". "
    # çok kısa metne (tek hece) en az ~0,6 sn yer ver; yoksa model konuşma yerine uğultu üretiyor
    tempo = min(tempo, len(b) * SN_HARF / EN_AZ_SN)
    return tempo * (1.3 if len(b) < 25 else 1.0) / factor


SN_HARF = 6.53 / 80                                     # örnek seste harf başına süre (ref.wav / REF_TEXT + ". ")
EN_AZ_SN = 0.6


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


def kisa_mi(text, kind):
    return kind in ("harf", "hece") or len(norm(text)) <= 3


def puan(text, kind, heard, ph, wav, sr):
    """İki denetçiyle puan (0..1) ve geçti mi.
    ph: fonem tanıma (ünlüler, sona eklenen ünlü); heard: harekeli Arapça tanıma (ünsüzler, kelimeler)."""
    import fonem
    dur = len(wav) / sr
    f = fonem.olc(text, ph)
    exp = 0.45 + 0.085 * len(norm(text))
    if dur < 0.2 or dur > exp * 2.2 + 0.6 or not ph or fonem.duz_mu(wav):
        return 0.0, False, f                                # boş, uğultu ya da fazladan ses eklenmiş
    ceza = (0.3 if f["fazla_son"] else 0) + (0.2 if f["eksik_son"] else 0)
    if kisa_mi(text, kind):
        ok = f["utam"] and f["csim"] >= 0.6 and not f["fazla_son"] and not f["eksik_son"]
        return 0.6 * f["usim"] + 0.4 * f["csim"] - ceza, ok, f
    s, ok = score(text, kind, heard, dur)
    ok = ok and f["usim"] >= 0.7 and f["csim"] >= 0.6 and not f["fazla_son"]
    return 0.55 * s + 0.3 * f["usim"] + 0.15 * f["csim"] - ceza, ok, f


def score(text, kind, heard, dur):
    """Harekeli tanımaya göre 0..1 arası puan ve geçti mi (uzun metinler)."""
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
    import fonem
    fon = fonem.Fonem(device)

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
        say = okunus(text, kind)                           # okunan: vakf + açık sükûn
        if k in index and os.path.exists(os.path.join(args.cikti, k + ".ogg")):
            continue
        try:
            best = None
            # kısa metinlerde (hece, harf) bütün tohumlar denenebilir; uzunlarda ilk üçü yeter
            for n_try, seed in enumerate(SEEDS if kisa_mi(say, kind) else SEEDS[:3], 1):
                # force_tashkeel=False: model harekeleri kendi düzelticisiyle değiştirmesin
                wav, sr, _ = tts.infer(ref_file=args.ref, ref_text=REF_TEXT, gen_text=say, file_wave=tmp, seed=seed,
                                       nfe_step=args.nfe, remove_silence=False, show_info=lambda *a, **k: None,
                                       speed=hiz(say), force_tashkeel=False)
                wav = np.asarray(wav, dtype=np.float32)
                heard = judge.hear(wav, sr) if not kisa_mi(say, kind) else ""
                ph = fon.duy(wav, sr)
                s, ok, f = puan(say, kind, heard, ph, wav, sr)
                if best is None or s > best[0]:
                    best = (s, ok, wav, sr, heard, n_try, f)
                if ok:
                    break
            s, ok, wav, sr, heard, n_try, f = best
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
            index[k] = {"t": text, "v": say, "k": kind, "ok": bool(ok), "s": round(float(s), 3), "n": n_try, "h": heard,
                        "f": f["fonem"], "u": f["usim"], "c": f["csim"], "fs": f["fazla_son"], "es": f["eksik_son"]}
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

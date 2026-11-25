"""GitHub Actions içinde çalışır: şifreli metinleri açar, sesleri üretir, çıktıyı şifreler.

Herkese açık kayıtlarda yalnızca ilerleme sayıları görünür; her şey (metinler, kütüphane
çıktıları, hatalar) şifreli günlüğe yazılır.

Kullanım: SES_ANAHTAR=<64 hex> python calistir.py --parca 3/20 --sure 18000
"""
import argparse, gzip, io, os, secrets, sys, tarfile


def aes():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    k = os.environ.get("SES_ANAHTAR", "").strip()
    if len(k) != 64:
        print("SES_ANAHTAR gizli değişkeni yok ya da hatalı (64 karakter olmalı).", flush=True)
        sys.exit(2)
    return AESGCM(bytes.fromhex(k))


def seal(a, data):
    iv = secrets.token_bytes(12)
    return iv + a.encrypt(iv, data, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parca", default="0/1")
    ap.add_argument("--sure", default="0")
    ap.add_argument("--girdi", default="girdi/metinler.enc")
    args = ap.parse_args()
    a = aes()
    i = args.parca.split("/")[0]
    os.makedirs("cikti", exist_ok=True)

    # ikinci tur gibi durumlarda farklı denemeler (girdi/tohumlar.txt: "41,53,67,...")
    if os.path.exists("girdi/tohumlar.txt"):
        os.environ["SES_TOHUMLAR"] = open("girdi/tohumlar.txt").read().strip()

    # 1) metinler
    raw = open(args.girdi, "rb").read()
    open("metinler.json", "wb").write(gzip.decompress(a.decrypt(raw[:12], raw[12:], None)))

    # 2) örnek ses: ref.wav (Arabic Speech Corpus, Nawar Halabi, CC BY 4.0) repoda duruyor
    if not os.path.exists("ref.wav"):
        print("ref.wav bulunamadı", flush=True)
        sys.exit(2)

    # 3) konsol = yalnız ilerleme; diğer her şey günlüğe
    progress = os.dup(1)
    os.environ["ILERLEME_FD"] = str(progress)
    log = open("gunluk.txt", "w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    status = 0
    try:
        import uret
        sys.argv = ["uret.py", "--metinler", "metinler.json", "--ref", "ref.wav", "--cikti", "ses_out",
                    "--parca", args.parca, "--sure", args.sure]
        uret.main()
    except BaseException as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        os.write(progress, f"HATA: {type(e).__name__}\n".encode())
        status = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()

    # 4) çıktıyı şifrele
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        if os.path.isdir("ses_out"):
            for name in sorted(os.listdir("ses_out")):
                if name.endswith((".ogg", ".json")):
                    t.add(os.path.join("ses_out", name), arcname=name)
    open(f"cikti/ses_{i}.enc", "wb").write(seal(a, buf.getvalue()))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        if os.path.isdir("ses_out/usta"):
            t.add("ses_out/usta", arcname="usta")
    open(f"cikti/usta_{i}.enc", "wb").write(seal(a, buf.getvalue()))
    log.flush()
    open(f"cikti/gunluk_{i}.enc", "wb").write(seal(a, gzip.compress(open("gunluk.txt", "rb").read())))
    n = len([x for x in os.listdir("ses_out") if x.endswith(".ogg")]) if os.path.isdir("ses_out") else 0
    os.write(progress, f"bitti: {n} ses\n".encode())
    sys.exit(status)


if __name__ == "__main__":
    main()

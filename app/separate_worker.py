"""Stimme von Musik/Geräuschen trennen – läuft in eigener venv (/opt/separator).

Modell: BS-RoFormer (model_bs_roformer_ep_317_sdr_12.9755.ckpt) über
python-audio-separator (MIT). Stimmen-SDR 12,9 – deutlich sauberer als
Demucs htdemucs: weniger Musikreste, weniger verlorene Höhen/Zischlaute.
Eigene venv, weil audio-separator eigene Torch-/ONNX-Versionen mitbringt.
Nur Standardbibliothek + audio_separator, keine Importe aus app/.

Aufruf: /opt/separator/bin/python app/separate_worker.py EINGABE.wav AUSGABE.wav MODELLORDNER
Gibt bei Erfolg den Pfad der Stimmspur aus (letzte Zeile).
"""

import os
import shutil
import sys

MODEL = os.environ.get("SEPARATOR_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")


def main() -> None:
    src, dst, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    from audio_separator.separator import Separator

    out_dir = os.path.dirname(os.path.abspath(dst))
    sep = Separator(output_dir=out_dir, model_file_dir=model_dir,
                    output_single_stem="Vocals", output_format="WAV")
    sep.load_model(model_filename=MODEL)
    files = sep.separate(src)
    if not files:
        sys.exit("audio-separator hat keine Stimmspur geliefert")
    vocals = files[0] if os.path.isabs(files[0]) else os.path.join(out_dir, files[0])
    shutil.move(vocals, dst)
    print(dst, flush=True)


if __name__ == "__main__":
    main()

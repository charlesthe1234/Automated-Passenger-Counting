"""Lightweight lossless ReID evidence writer subprocess.

The parent communicates through a trusted pickle stream over stdin/stdout.
Keeping this module separate avoids importing the tracking, Torch, YOLO, and
ReID stacks in the evidence process.
"""

import hashlib
import os
import pickle
import sys
from pathlib import Path

import cv2


def _crop_digest(crop):
    digest = hashlib.sha256()
    digest.update(str(crop.shape).encode("ascii"))
    digest.update(str(crop.dtype).encode("ascii"))
    digest.update(crop.tobytes())
    return digest.hexdigest()


def _write_evidence(task):
    output_path = Path(task["output_path"])
    temporary_path = output_path.parent / f".{output_path.stem}.{os.getpid()}.tmp.png"
    try:
        crop = task["crop"]
        if crop is None or crop.size == 0:
            raise ValueError("empty evidence crop")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        digest = _crop_digest(crop)
        # PNG compression changes file size and encoding time, not pixel
        # quality. Level 0 is the fastest lossless representation.
        if not cv2.imwrite(
            str(temporary_path),
            crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError("cv2.imwrite returned False")
        os.replace(temporary_path, output_path)
        return {"ok": True, "digest": digest}
    except Exception as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": False, "error": str(exc)}


def main():
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        try:
            task = pickle.load(input_stream)
        except EOFError:
            return
        if task is None:
            return
        pickle.dump(_write_evidence(task), output_stream, protocol=pickle.HIGHEST_PROTOCOL)
        output_stream.flush()


if __name__ == "__main__":
    main()

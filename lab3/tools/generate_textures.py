"""Rebuild the original, seamless PNG assets; requires only Python's standard library."""
from pathlib import Path
import math
import struct
import zlib

OUTPUT = Path(__file__).resolve().parents[1] / "textures"


def save(name, pixel, width=512, height=256):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(max(0, min(255, round(v))) for v in pixel(x / width, y / height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    (OUTPUT / name).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                               chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def porcelain(u, v):
    x, y = u * 12 % 1, v * 6 % 1
    border = min(x, y, 1 - x, 1 - y) < 0.045
    if border:
        return (204, 155, 64)
    flower = math.cos(2 * math.pi * x) * math.cos(2 * math.pi * y)
    if flower > 0.38:
        return (32 + 16 * flower, 90 + 30 * flower, 156 + 38 * flower)
    return (216, 230, 235)


def porcelain_specular(u, v):
    x, y = u * 12 % 1, v * 6 % 1
    return (235,) * 3 if min(x, y, 1 - x, 1 - y) < 0.045 else (65,) * 3


def circuit_mask(u, v):
    x, y = u * 16 % 1, v * 8 % 1
    trace = (abs(y - 0.25) < 0.025 and x < 0.55) or (abs(x - 0.5) < 0.025 and y < 0.78)
    trace |= abs(y - 0.75) < 0.025 and x > 0.45
    pad = math.hypot(x - 0.15, y - 0.25) < 0.07 or math.hypot(x - 0.85, y - 0.75) < 0.07
    return trace or pad


def circuit(u, v):
    if circuit_mask(u, v):
        return (48, 185, 160)
    if 0.20 < u * 16 % 1 < 0.75 and 0.37 < v * 8 % 1 < 0.62:
        return (48, 65, 78)
    return (14, 28, 42)


def circuit_specular(u, v):
    return (220,) * 3 if circuit_mask(u, v) else (30,) * 3


def circuit_emissive(u, v):
    if circuit_mask(u, v):
        wave = 0.65 + 0.35 * math.sin(2 * math.pi * (u * 3 + v * 2)) ** 2
        return (20 * wave, 220 * wave, 170 * wave)
    return (0, 0, 0)


def woven(u, v):
    check = (int(u * 16) + int(v * 8)) % 2
    base = (182, 82, 45) if check else (237, 202, 139)
    thread = 0.78 + 0.22 * math.sin(2 * math.pi * u * 256) * math.sin(2 * math.pi * v * 128)
    return tuple(value * thread for value in base)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, function in [("porcelain", porcelain), ("porcelain_specular", porcelain_specular),
                           ("circuit", circuit), ("circuit_specular", circuit_specular),
                           ("circuit_emissive", circuit_emissive), ("woven", woven)]:
        save(name + ".png", function)
    save("black.png", lambda u, v: (0, 0, 0), 1, 1)

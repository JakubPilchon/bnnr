#!/usr/bin/env python3
"""Build README XAI comparison figure(s) from benchmark run overlays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/assets/benchmark-xai-comparison.png"
OPTIONS_DIR = REPO / "docs/assets/benchmark-xai-options"
RESULTS = REPO / "benchmarks/results.json"
SEED = 44
SCALE = 10
CARD_PAD = 20
GAP = 20
OUTER_PAD = 28
ACCENT_BAR = 4

# BNNR brand (dashboard / analyze report)
BG = (8, 8, 16)
CARD = (19, 19, 34)
CARD_BORDER = (42, 42, 68)
FG = (241, 245, 249)
MUTED = (148, 163, 184)
ACCENT = (240, 160, 105)
ACCENT_SOFT = (245, 184, 136)

CONDITIONS = [
    ("Without BNNR", "no_bnnr", "OptiCAM overlay"),
    ("RandAugment", "randaugment", "OptiCAM overlay"),
    ("BNNR branch search", "bnnr_branch_search", "OptiCAM overlay"),
]


def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ):
        path = Path("/usr/share/fonts/truetype/dejavu") / name
        if not path.exists():
            path = Path("/usr/share/fonts/truetype/inter") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_cifar_val_rgb(val_idx: int) -> Image.Image:
    from torchvision import datasets, transforms

    ds = datasets.CIFAR10(
        str(REPO / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    tensor, _ = ds[val_idx]
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _scale_nearest(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.Resampling.NEAREST)


def _load_context(seed: int = SEED) -> tuple[dict, dict[str, Path], list[int]]:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    run_dirs: dict[str, Path] = {}
    for _, cond, _ in CONDITIONS:
        runs = [r for r in data["runs"] if r.get("condition") == cond and r.get("seed") == seed]
        if not runs:
            raise SystemExit(f"No run for {cond!r} seed {seed} in {RESULTS}")
        run_dirs[cond] = REPO / runs[-1]["run_dir"]
    manifest = json.loads((run_dirs["no_bnnr"] / "xai/manifest.json").read_text(encoding="utf-8"))
    val_indices = list(manifest["sample_indices"])
    return data, run_dirs, val_indices


def _resolve_attention_index(pick: str, val_indices: list[int]) -> int:
    if pick.isdigit() and int(pick) in val_indices:
        return val_indices.index(int(pick))
    m = re.match(r"(?:val|idx)?(\d+)$", pick, re.I)
    if m:
        vi = int(m.group(1))
        if vi in val_indices:
            return val_indices.index(vi)
    if pick.startswith("attn") and pick[4:].isdigit():
        return int(pick[4:])
    raise SystemExit(f"Unknown pick {pick!r}. Use val index ({val_indices}) or attn0–7.")


def _make_panel(
    *,
    title: str,
    line2: str,
    line3: str,
    image: Image.Image,
    accent: tuple[int, int, int] = ACCENT,
) -> Image.Image:
    title_font = _font(True, 16)
    sub_font = _font(False, 13)
    meta_font = _font(False, 12)

    img = _scale_nearest(image, SCALE)
    inner_w = img.width + 2
    inner_h = img.height + 2

    # Measure header block
    probe = Image.new("RGB", (1, 1))
    pdraw = ImageDraw.Draw(probe)
    header_h = (
        CARD_PAD
        + (pdraw.textbbox((0, 0), title, font=title_font)[3] - pdraw.textbbox((0, 0), title, font=title_font)[1])
        + 6
        + (pdraw.textbbox((0, 0), line2, font=sub_font)[3] - pdraw.textbbox((0, 0), line2, font=sub_font)[1])
        + 4
        + (pdraw.textbbox((0, 0), line3, font=meta_font)[3] - pdraw.textbbox((0, 0), line3, font=meta_font)[1])
        + 14
    )
    panel_w = inner_w + CARD_PAD * 2
    panel_h = ACCENT_BAR + header_h + inner_h + CARD_PAD

    panel = Image.new("RGB", (panel_w, panel_h), CARD)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_w, ACCENT_BAR), fill=accent)
    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=CARD_BORDER)

    y = ACCENT_BAR + CARD_PAD
    draw.text((CARD_PAD, y), title, fill=FG, font=title_font)
    y += draw.textbbox((CARD_PAD, y), title, font=title_font)[3] - draw.textbbox((CARD_PAD, y), title, font=title_font)[1] + 6
    draw.text((CARD_PAD, y), line2, fill=ACCENT_SOFT, font=sub_font)
    y += draw.textbbox((CARD_PAD, y), line2, font=sub_font)[3] - draw.textbbox((CARD_PAD, y), line2, font=sub_font)[1] + 4
    draw.text((CARD_PAD, y), line3, fill=MUTED, font=meta_font)
    y += draw.textbbox((CARD_PAD, y), line3, font=meta_font)[3] - draw.textbbox((CARD_PAD, y), line3, font=meta_font)[1] + 14

    frame_x = CARD_PAD
    draw.rectangle((frame_x, y, frame_x + inner_w - 1, y + inner_h - 1), outline=CARD_BORDER)
    panel.paste(img, (frame_x + 1, y + 1))
    return panel


def build_comparison(
    data: dict,
    run_dirs: dict[str, Path],
    val_indices: list[int],
    attention_index: int,
    *,
    include_original: bool = True,
) -> Image.Image:
    val_idx = val_indices[attention_index]
    title_font = _font(True, 20)
    sub_font = _font(False, 14)

    panels: list[Image.Image] = []

    if include_original:
        raw = _load_cifar_val_rgb(val_idx)
        panels.append(
            _make_panel(
                title="Original input",
                line2="No XAI overlay",
                line3=f"CIFAR-10 test index {val_idx}",
                image=raw,
                accent=(129, 140, 248),  # soft indigo — distinct from peach XAI panels
            )
        )

    for label, cond, kind in CONDITIONS:
        acc = float(next(r for r in data["runs"] if r["condition"] == cond and r["seed"] == SEED)["val_metric"]) * 100
        overlay = run_dirs[cond] / "xai" / f"attention_{attention_index}.png"
        panels.append(
            _make_panel(
                title=label,
                line2=f"{acc:.1f}% validation accuracy",
                line3=kind,
                image=Image.open(overlay).convert("RGB"),
                accent=ACCENT,
            )
        )

    row_w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    row_h = max(p.height for p in panels)

    # Header
    header_title = "Where the model looks — same validation image"
    header_sub = f"CIFAR-10 · test index {val_idx} · OptiCAM · seed {SEED} · demo CNN benchmark"

    probe = Image.new("RGB", (1, 1))
    pdraw = ImageDraw.Draw(probe)
    title_h = pdraw.textbbox((0, 0), header_title, font=title_font)[3] - pdraw.textbbox((0, 0), header_title, font=title_font)[1]
    sub_h = pdraw.textbbox((0, 0), header_sub, font=sub_font)[3] - pdraw.textbbox((0, 0), header_sub, font=sub_font)[1]
    header_block_h = title_h + 8 + sub_h + 20

    canvas_w = row_w + OUTER_PAD * 2
    canvas_h = OUTER_PAD + header_block_h + row_h + OUTER_PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    # Top brand accent line
    draw.rectangle((OUTER_PAD, OUTER_PAD, canvas_w - OUTER_PAD, OUTER_PAD + 3), fill=ACCENT)

    tx = OUTER_PAD
    ty = OUTER_PAD + 16
    draw.text((tx, ty), header_title, fill=FG, font=title_font)
    ty += title_h + 8
    draw.text((tx, ty), header_sub, fill=MUTED, font=sub_font)

    row_y = OUTER_PAD + header_block_h
    x = OUTER_PAD
    for p in panels:
        canvas.paste(p, (x, row_y + (row_h - p.height) // 2))
        x += p.width + GAP

    return canvas


def write_all_options(data: dict, run_dirs: dict[str, Path], val_indices: list[int]) -> None:
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for ai in range(len(val_indices)):
        canvas = build_comparison(data, run_dirs, val_indices, ai)
        vi = val_indices[ai]
        path = OPTIONS_DIR / f"comparison_val{vi}_attn{ai}.png"
        canvas.save(path, optimize=True)
        print(f"  {path.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-options", action="store_true", help="Write 8 variants to docs/assets/benchmark-xai-options/")
    parser.add_argument("--pick", type=str, help="Val index (e.g. 127) → benchmark-xai-comparison.png")
    parser.add_argument("--attention-index", type=int, help="Direct attention index 0–7 for main output")
    args = parser.parse_args()

    data, run_dirs, val_indices = _load_context()

    if args.all_options:
        print("Writing options:")
        write_all_options(data, run_dirs, val_indices)
        return

    if args.pick:
        ai = _resolve_attention_index(args.pick, val_indices)
    else:
        ai = args.attention_index if args.attention_index is not None else 1

    canvas = build_comparison(data, run_dirs, val_indices, ai)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print(f"Wrote {OUT.relative_to(REPO)} ({canvas.width}×{canvas.height}, val idx {val_indices[ai]})")


if __name__ == "__main__":
    main()

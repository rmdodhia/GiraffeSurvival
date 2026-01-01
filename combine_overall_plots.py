"""Combine giraffe growth plots into a multi-page PDF.

Each page is a 2x2 grid containing up to 3 plots for one measure:
    1) overall
    2) by sex
    3) sex-unknown overall

Pages are generated for: TH, ossicone, neck, foreleg.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

from PIL import Image


def load_images(paths: List[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except OSError:
            print(f"Skipping unreadable image: {path}", file=sys.stderr)
    return images


def ensure_image_count(images: List[Image.Image]) -> List[Image.Image]:
    if not images:
        return images

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    while len(images) < 4:
        images.append(Image.new("RGB", (max_width, max_height), color="white"))

    return images[:4]


def build_grid(images: List[Image.Image], columns: int = 2) -> Image.Image:
    if not images:
        raise ValueError("No images provided to build_grid")

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    rows = 2  # fixed 2x2 grid
    sheet = Image.new("RGB", (columns * max_width, rows * max_height), color="white")

    for idx, img in enumerate(images):
        row, col = divmod(idx, columns)
        if row >= rows:
            break
        resized = img.resize((max_width, max_height))
        sheet.paste(resized, (col * max_width, row * max_height))

    return sheet


def _blank_like(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        return Image.new("RGB", (1000, 800), color="white")
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    return Image.new("RGB", (max_width, max_height), color="white")


def _load_or_blank(path: Path, blank: Image.Image) -> Image.Image:
    if not path.exists():
        return blank
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except OSError:
        print(f"Skipping unreadable image: {path}", file=sys.stderr)
        return blank


def _page_paths_for_label(graph_dir: Path, label: str) -> list[Path]:
    # plot_growth_curve_overall -> {label}_overall.png
    # plot_growth_curve_by_sex -> {label}_by_sex.png
    # unknown subset uses df_label=f"{label}_unknown" -> {label}_unknown_overall.png
    return [
        graph_dir / f"{label}_overall.png",
        graph_dir / f"{label}_by_sex.png",
        graph_dir / f"{label}_unknown_overall.png",
    ]


def main() -> None:
    graph_dir = Path("Graph")
    output_path = graph_dir / "overall_grids_by_measure.pdf"

    if not graph_dir.exists():
        print("Graph directory not found. Nothing to do.")
        return

    measures = [
        ("th", "wild_th"),
        ("ossicone", "wild_ossicone"),
        ("neck", "wild_neck"),
        ("leg", "wild_leg"),
    ]

    # Create a blank tile sized to the first available image.
    any_pngs = sorted(graph_dir.glob("*.png"))
    seed_images = load_images(any_pngs[:10])
    blank = _blank_like(seed_images)

    pages: list[Image.Image] = []
    for _, label in measures:
        paths = _page_paths_for_label(graph_dir, label)
        tiles = [_load_or_blank(p, blank) for p in paths]
        # Ensure exactly 4 tiles (2x2), leaving one blank spot.
        while len(tiles) < 4:
            tiles.append(blank)
        tiles = tiles[:4]
        pages.append(build_grid(tiles, columns=2))

    if not pages:
        print("No pages generated.")
        return

    # Save as a multi-page PDF.
    pages[0].save(output_path, save_all=True, append_images=pages[1:])
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()

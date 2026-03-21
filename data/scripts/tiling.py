"""
Shared tiling geometry used by BOTH training data prep and inference.

This module is the single source of truth for:
- Tile coordinate generation
- Edge anchoring
- Bbox clipping and inclusion rules

HARD INVARIANT: prepare_tiles.py and run_onnx_tiled.py must both use
these functions. Geometry drift between train and infer breaks the pipeline.
"""
import math


def generate_tile_coords(img_w, img_h, tile_size, stride):
    """
    Generate tile (x, y) origins for an image.

    Includes edge-anchored tiles to ensure full right/bottom coverage.
    Deduplicates if edge tile overlaps with grid tile.

    Returns list of (x, y) tuples.
    """
    tiles = set()

    # Regular grid
    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            tiles.add((x, y))

    # Right-edge anchored column
    if img_w > tile_size:
        right_x = img_w - tile_size
        for y in range(0, img_h - tile_size + 1, stride):
            tiles.add((right_x, y))

    # Bottom-edge anchored row
    if img_h > tile_size:
        bottom_y = img_h - tile_size
        for x in range(0, img_w - tile_size + 1, stride):
            tiles.add((x, bottom_y))

    # Bottom-right corner
    if img_w > tile_size and img_h > tile_size:
        tiles.add((img_w - tile_size, img_h - tile_size))

    # Handle images smaller than tile_size
    if img_w <= tile_size or img_h <= tile_size:
        tiles.add((0, 0))

    return sorted(tiles)


def clip_bbox_to_tile(bbox, tile_x, tile_y, tile_size,
                      min_area_fraction=0.5, min_pixel_dim=10):
    """
    Clip a COCO bbox [x, y, w, h] to a tile's boundaries.

    Returns (clipped_local_bbox, include) where:
    - clipped_local_bbox is [local_x, local_y, local_w, local_h] in tile coordinates
    - include is True if the bbox passes inclusion criteria

    Inclusion criteria:
    - Clipped area >= min_area_fraction of original area
    - Clipped width >= min_pixel_dim
    - Clipped height >= min_pixel_dim
    """
    bx, by, bw, bh = bbox
    original_area = bw * bh

    if original_area <= 0:
        return None, False

    # Tile bounds
    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + tile_size, tile_y + tile_size

    # Bbox bounds
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(bx, tx1)
    iy1 = max(by, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    clipped_w = ix2 - ix1
    clipped_h = iy2 - iy1

    if clipped_w <= 0 or clipped_h <= 0:
        return None, False

    clipped_area = clipped_w * clipped_h

    # Check inclusion criteria
    if clipped_area / original_area < min_area_fraction:
        return None, False
    if clipped_w < min_pixel_dim:
        return None, False
    if clipped_h < min_pixel_dim:
        return None, False

    # Convert to tile-local coordinates
    local_x = ix1 - tile_x
    local_y = iy1 - tile_y

    return [local_x, local_y, clipped_w, clipped_h], True


def bbox_to_yolo(bbox, tile_size):
    """Convert local [x, y, w, h] bbox to YOLO [cx, cy, nw, nh] normalized by tile_size."""
    x, y, w, h = bbox
    cx = (x + w / 2) / tile_size
    cy = (y + h / 2) / tile_size
    nw = w / tile_size
    nh = h / tile_size
    return cx, cy, nw, nh


def map_tile_detection_to_image(det_bbox, tile_x, tile_y):
    """
    Map a detection bbox from tile coordinates to full-image coordinates.
    det_bbox: [x, y, w, h] in tile coordinates
    Returns: [x, y, w, h] in full-image coordinates
    """
    x, y, w, h = det_bbox
    return [tile_x + x, tile_y + y, w, h]

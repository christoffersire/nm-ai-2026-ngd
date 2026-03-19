"""
Scorer edge-case tests.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from scoring.scorer import compute_iou, match_predictions_to_gt, compute_ap, compute_map, hybrid_score


def make_pred(img_id, cat_id, bbox, score):
    return {"image_id": img_id, "category_id": cat_id, "bbox": bbox, "score": score}


def make_gt(img_id, cat_id, bbox):
    return {"image_id": img_id, "category_id": cat_id, "bbox": bbox}


def test_iou_perfect():
    assert abs(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6


def test_iou_no_overlap():
    assert compute_iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0


def test_iou_partial():
    iou = compute_iou([0, 0, 10, 10], [5, 0, 10, 10])
    # Overlap: 5x10=50, union: 200-50=150
    assert abs(iou - 50 / 150) < 1e-6


def test_perfect_match():
    """All predictions perfectly match GT."""
    preds = [make_pred(1, 5, [0, 0, 10, 10], 0.9)]
    gt = [make_gt(1, 5, [0, 0, 10, 10])]

    result = hybrid_score(preds, gt)
    assert result["detection_mAP"] > 0.99
    assert result["classification_mAP"] > 0.99
    assert result["hybrid_score"] > 0.99


def test_wrong_class_right_box():
    """Detection should score, classification should not."""
    preds = [make_pred(1, 5, [0, 0, 10, 10], 0.9)]
    gt = [make_gt(1, 7, [0, 0, 10, 10])]

    result = hybrid_score(preds, gt)
    assert result["detection_mAP"] > 0.99
    assert result["classification_mAP"] < 0.01


def test_duplicate_boxes():
    """Second prediction on same GT should be FP, but AP stays 1.0 because
    the highest-confidence pred is TP and covers all GT (COCO AP property)."""
    preds = [
        make_pred(1, 5, [0, 0, 10, 10], 0.9),
        make_pred(1, 5, [0, 0, 10, 10], 0.8),  # duplicate, FP
    ]
    gt = [make_gt(1, 5, [0, 0, 10, 10])]

    result = hybrid_score(preds, gt)
    # With 1 GT, highest-conf pred is TP → AP=1.0 (low-conf FP doesn't hurt)
    assert result["detection_mAP"] > 0.99


def test_duplicate_boxes_hurts_when_fp_is_higher_conf():
    """When the FP has higher confidence than the TP, AP drops."""
    preds = [
        make_pred(1, 5, [50, 50, 10, 10], 0.95),  # FP (no matching GT)
        make_pred(1, 5, [0, 0, 10, 10], 0.8),      # TP
    ]
    gt = [make_gt(1, 5, [0, 0, 10, 10])]

    result = hybrid_score(preds, gt)
    # First pred is FP (precision=0), second is TP → AP < 1.0
    assert result["detection_mAP"] < 1.0


def test_low_iou():
    """Box below IoU threshold should not match."""
    preds = [make_pred(1, 5, [0, 0, 10, 10], 0.9)]
    gt = [make_gt(1, 5, [50, 50, 10, 10])]  # no overlap

    result = hybrid_score(preds, gt)
    assert result["detection_mAP"] < 0.01
    assert result["classification_mAP"] < 0.01


def test_empty_predictions():
    """No predictions — score should be 0."""
    preds = []
    gt = [make_gt(1, 5, [0, 0, 10, 10])]

    result = hybrid_score(preds, gt)
    assert result["detection_mAP"] == 0.0
    assert result["classification_mAP"] == 0.0


def test_empty_gt():
    """No GT — predictions are all FP, but no GT means AP=0."""
    preds = [make_pred(1, 5, [0, 0, 10, 10], 0.9)]
    gt = []

    result = hybrid_score(preds, gt)
    assert result["detection_mAP"] == 0.0
    assert result["classification_mAP"] == 0.0


def test_mixed():
    """Some correct, some wrong, some missing."""
    preds = [
        make_pred(1, 5, [0, 0, 10, 10], 0.95),     # correct
        make_pred(1, 7, [20, 20, 10, 10], 0.85),    # wrong class, right box
        make_pred(1, 5, [80, 80, 10, 10], 0.75),    # no matching GT (FP)
    ]
    gt = [
        make_gt(1, 5, [0, 0, 10, 10]),
        make_gt(1, 7, [20, 20, 10, 10]),
        make_gt(1, 3, [50, 50, 10, 10]),   # missed (no prediction)
    ]

    result = hybrid_score(preds, gt)
    # Detection: 2 TP out of 3 GT, 1 FP
    assert 0 < result["detection_mAP"] < 1.0
    # Classification: 2 TP (class matches), 1 FP, 1 FN
    assert 0 < result["classification_mAP"] < 1.0


def test_multiple_gt_same_class_overlapping():
    """Multiple GT objects of same class — high-conf preds match both GT, low-conf is FP.
    Since TPs come first in score order, AP = 1.0 (same COCO AP property)."""
    preds = [
        make_pred(1, 5, [0, 0, 10, 10], 0.95),
        make_pred(1, 5, [12, 0, 10, 10], 0.85),
        make_pred(1, 5, [6, 0, 10, 10], 0.75),  # FP, but lower conf
    ]
    gt = [
        make_gt(1, 5, [0, 0, 10, 10]),
        make_gt(1, 5, [12, 0, 10, 10]),
    ]

    result = hybrid_score(preds, gt)
    # Both GT matched by higher-conf preds → AP = 1.0
    assert result["detection_mAP"] > 0.99
    assert result["classification_mAP"] > 0.99


def test_multiple_gt_same_class_with_misses():
    """Crowded shelf: some GT missed entirely."""
    preds = [
        make_pred(1, 5, [0, 0, 10, 10], 0.95),
    ]
    gt = [
        make_gt(1, 5, [0, 0, 10, 10]),
        make_gt(1, 5, [12, 0, 10, 10]),  # missed
        make_gt(1, 5, [24, 0, 10, 10]),  # missed
    ]

    result = hybrid_score(preds, gt)
    # 1 TP out of 3 GT → AP should be less than 1.0
    assert 0 < result["detection_mAP"] < 1.0


if __name__ == "__main__":
    tests = [
        test_iou_perfect,
        test_iou_no_overlap,
        test_iou_partial,
        test_perfect_match,
        test_wrong_class_right_box,
        test_duplicate_boxes,
        test_duplicate_boxes_hurts_when_fp_is_higher_conf,
        test_low_iou,
        test_empty_predictions,
        test_empty_gt,
        test_mixed,
        test_multiple_gt_same_class_overlapping,
        test_multiple_gt_same_class_with_misses,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")

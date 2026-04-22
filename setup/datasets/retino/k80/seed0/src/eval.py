def evaluate(coco_gt, coco_dt):
    """
    Evaluate classification predictions in a COCO-ish format.

    Expected prediction format (coco_results passed to coco_gt.loadRes):
        [{"image_id": int, "category_id": int, "score": float}, ...]

    Ground truth:
        coco_gt must have annotations; we assume one label per image
        (if multiple anns, we take the first category_id).

    Returns:
        dict with accuracy + counts
    """
    # ---- build GT: image_id -> category_id ----
    gt = {}
    for img_id in coco_gt.getImgIds():
        anns = coco_gt.imgToAnns.get(img_id, [])
        if not anns:
            continue
        gt[img_id] = int(anns[0]["category_id"])  # simple: first annotation

    if not gt:
        return {"error": "No ground-truth annotations found in coco_gt."}

    # ---- pull predictions from coco_dt ----
    # coco_dt is a COCO object created by loadRes; its "annotations" are your predictions
    preds_raw = coco_dt.dataset.get("annotations", [])
    if not preds_raw:
        return {"error": "No predictions found in coco_dt (empty annotations list)."}

    # image_id -> list of (score, category_id)
    pred_by_img = {}
    for p in preds_raw:
        img_id = int(p["image_id"])
        cat_id = int(p["category_id"])
        score = float(p.get("score", 1.0))
        pred_by_img.setdefault(img_id, []).append((score, cat_id))

    # ---- compute top-1 (and top-5 if possible) ----
    correct_top1 = 0
    correct_top5 = 0
    evaluated = 0
    missing_pred = 0

    for img_id, gt_cat in gt.items():
        if img_id not in pred_by_img:
            missing_pred += 1
            continue

        evaluated += 1
        ranked = sorted(pred_by_img[img_id], key=lambda x: x[0], reverse=True)

        top1_cat = ranked[0][1]
        if top1_cat == gt_cat:
            correct_top1 += 1

        top5_cats = [c for _, c in ranked[:5]]
        if gt_cat in top5_cats:
            correct_top5 += 1

    if evaluated == 0:
        return {
            "error": "No overlapping image_ids between ground truth and predictions.",
            "num_gt_images": len(gt),
            "num_pred_images": len(pred_by_img),
        }

    acc1 = correct_top1 / evaluated
    acc5 = correct_top5 / evaluated

    return {
        "num_gt_images": len(gt),
        "num_pred_images": len(pred_by_img),
        "num_evaluated": evaluated,
        "num_missing_predictions": missing_pred,
        "top1_accuracy": acc1,
        "top5_accuracy": acc5,
    }
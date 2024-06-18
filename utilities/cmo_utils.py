import numpy as np


def cut_mix(images_orig, labels_orig, images_cmo, labels_cmo, args):
    lam = np.random.beta(args.cmo_beta * 2, args.cmo_beta)

    bbx1, bby1, bbx2, bby2 = rand_bbox_withcenter(
        images_orig.size(), lam, images_orig.shape[2] // 2, images_orig.shape[3] // 2
    )
    # randomly shift where the cutout is placed within the image
    shiftx = np.random.randint((images_orig.size(2) - (bbx2 - bbx1)) // 2)
    shifty = np.random.randint((images_orig.size(3) - (bby2 - bby1)) // 2)
    # select sign
    shift_signx = np.random.randint(2) * 2 - 1
    shift_signy = np.random.randint(2) * 2 - 1

    images_orig[
        :,
        :,
        bbx1 + shiftx * shift_signx : bbx2 + shiftx * shift_signx,
        bby1 + shift_signy * shifty : bby2 + shift_signy * shifty,
    ] = images_cmo[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1) / (images_orig.size(2) * images_orig.size(3))
    )
    # mix labels
    labels_orig = labels_orig * lam + labels_cmo * (1 - lam)
    return images_orig, labels_orig


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

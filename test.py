"""Evaluation for BA-Net: shared ``evaluate()`` + standalone ``test.py`` entry."""
import argparse
import sys
from pathlib import Path

import torch

from utils.losses import joint_loss1
from utils.metrics import compute_hd95, dice_coef, jaccard_similarity, recall_precision
from models.banet import Proposed

REPO_ROOT = Path(__file__).resolve().parent


def evaluate(loader, teacher, device, with_loss=True, with_standard_metrics=True, with_hd95=False):
    teacher.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    running_recall = 0.0
    running_precision = 0.0
    running_hd95 = 0.0

    with torch.no_grad():
        for images, _, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            t0, _, _, _, _ = teacher(images)

            if with_loss:
                loss = joint_loss1(masks, t0)
                running_loss += loss.item()

            preds = (t0 > 0.5).float()

            if with_standard_metrics:
                running_dice += dice_coef(masks, preds).item()
                running_jaccard += jaccard_similarity(masks, preds).item()
                recall, precision = recall_precision(masks, preds)
                running_recall += recall.item()
                running_precision += precision.item()

            if with_hd95:
                running_hd95 += compute_hd95(pred=preds, target=masks)

    n = max(len(loader), 1)
    return (
        running_loss / n,
        running_dice / n,
        running_jaccard / n,
        running_recall / n,
        running_precision / n,
        running_hd95 / n,
    )


def main(checkpoint=None, dataset_name='OTU', batch_size=4, num_workers=0,
         pin_memory=False, labeled_ratio=0.1, device_id='cuda:0', annotation_file=None):
    from data.loader import get_dataloaders

    device = torch.device(device_id)
    checkpoint = Path(checkpoint) if checkpoint else (REPO_ROOT / "checkpoints" / "proposed.pth")
    print("Device:", device)
    print("Checkpoint:", checkpoint)

    model = Proposed().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    _, _, _, test_loader = get_dataloaders(
        dataset_name, batch_size, num_workers, pin_memory, labeled_ratio,
        annotation_file=annotation_file,
    )

    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(
        test_loader, model, device,
        with_loss=True, with_standard_metrics=True, with_hd95=True,
    )
    print(f"  Test Dice Coef: {test_dice:.4f}")
    print(f"  Test Jaccard Similarity: {test_jaccard:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test HD95: {test_hd95:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BA-Net (Proposed) on the test split")
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "proposed.pth"))
    parser.add_argument("--dataset_name", type=str, default='OTU')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--device_id", type=str, default='cuda:0')
    parser.add_argument("--annotation_file", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))

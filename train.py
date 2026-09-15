"""Two-stage Mean-Teacher training for BA-Net (Proposed).

Stage 1 (pre-train): labeled data only, supervised deep-supervision loss
    + EMA consistency loss.
Stage 2 (self-train): labeled + unlabeled data, supervised + consistency
    + pseudo-label boundary loss from the frozen best pre-train teacher.
"""
import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.optim as optim

from data import get_dataloaders
from loss import MSE_loss, muti_bce_loss_fusion, unlabeled_loss
from model import Proposed
from ramp import sigmoid_rampup
from test import evaluate

REPO_ROOT = Path(__file__).resolve().parent
global_step = 0


def get_ema_alpha(step, rampup_length, start, end):
    return start + (end - start) * sigmoid_rampup(current=step, rampup_length=rampup_length)


def update_ema_variables(model, ema_model, step, rampup_length, start, end, fixed_alpha=None):
    alpha = fixed_alpha if fixed_alpha is not None else get_ema_alpha(step, rampup_length, start, end)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def pre_train_one_epoch(epoch, student, teacher, labeled_train_loader, optimizer,
                        device, rampup_length, max_lambda, start_ema_coef, end_ema_coef):
    global global_step

    student.train()
    teacher.train()

    for student_images, teacher_images, masks, is_labeled in labeled_train_loader:
        LAMBDA = sigmoid_rampup(global_step, rampup_length) * max_lambda

        student_images = student_images.to(device)
        teacher_images = teacher_images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        s0, s1, s2, s3, s4 = student(student_images)
        t0, t1, t2, t3, t4 = teacher(teacher_images)

        loss1 = muti_bce_loss_fusion(s0, s1, s2, s3, s4, masks)
        loss2 = MSE_loss(s0, t0) + MSE_loss(s1, t1) + MSE_loss(s2, t2) + MSE_loss(s3, t3) + MSE_loss(s4, t4)

        loss = loss1 + LAMBDA * loss2

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(student, teacher, global_step, rampup_length, start_ema_coef, end_ema_coef)


def self_train_one_epoch(epoch, student, teacher, pseudo_label_generator, train_loader,
                         optimizer, device, rampup_length, max_lambda, max_beta,
                         start_ema_coef, end_ema_coef):
    global global_step

    student.train()
    teacher.train()
    pseudo_label_generator.train()

    for student_images, teacher_images, masks, is_labeled in train_loader:
        LAMBDA = sigmoid_rampup(global_step, rampup_length) * max_lambda
        BETA = sigmoid_rampup(global_step, rampup_length) * max_beta

        student_images_lab = student_images[is_labeled].to(device)
        teacher_images_lab = teacher_images[is_labeled].to(device)
        masks = masks[is_labeled].to(device)

        student_images_unlab = student_images[~is_labeled].to(device)
        teacher_images_unlab = teacher_images[~is_labeled].to(device)

        optimizer.zero_grad()

        s0, s1, s2, s3, s4 = student(student_images_lab)
        s0_un, s1_un, s2_un, s3_un, s4_un = student(student_images_unlab)

        t0, t1, t2, t3, t4 = teacher(teacher_images_lab)
        t0_un, t1_un, t2_un, t3_un, t4_un = teacher(teacher_images_unlab)

        target0_un, target1_un, _, _, _ = pseudo_label_generator(teacher_images_unlab)

        loss1 = muti_bce_loss_fusion(s0, s1, s2, s3, s4, masks)

        loss2 = (
            MSE_loss(s0, t0) + MSE_loss(s1, t1) + MSE_loss(s2, t2) +
            MSE_loss(s3, t3) + MSE_loss(s4, t4) +
            MSE_loss(s0_un, t0_un) + MSE_loss(s1_un, t1_un) +
            MSE_loss(s2_un, t2_un) + MSE_loss(s3_un, t3_un) +
            MSE_loss(s4_un, t4_un)
        )

        loss3 = unlabeled_loss(s0_un, s1_un, torch.round(target0_un), torch.round(target1_un))

        loss = loss1 + LAMBDA * loss2 + BETA * loss3

        loss.backward()
        optimizer.step()

        global_step += 1
        update_ema_variables(student, teacher, global_step, rampup_length, start_ema_coef, end_ema_coef)


def main(image_size=256, batch_size=4, num_workers=0, pin_memory=False,
         labeled_ratio=0.1, dataset_name='OTU', pre_epochs=50, epochs=50,
         max_lambda=1.0, max_beta=1.0, start_ema_coef=0.99, end_ema_coef=0.999,
         learning_rate=0.001, device_id='cuda:0',
         best_model_path=None, annotation_file=None):
    global global_step

    device = torch.device(device_id)
    best_model_path = Path(best_model_path) if best_model_path else (REPO_ROOT / "weight" / "proposed.pth")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    print("Proposed")

    student = Proposed().to(device)
    teacher = copy.deepcopy(student).to(device)
    pseudo_label_generator = copy.deepcopy(student).to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(
        dataset_name, batch_size, num_workers, pin_memory, labeled_ratio,
        annotation_file=annotation_file,
    )

    rampup_length = pre_epochs * len(labeled_train_loader) + epochs * len(train_loader)
    global_step = 0

    # ================= Pre-train =================
    print("\n--- Pre-train ---")
    best_loss = float('inf')
    for epoch in range(pre_epochs):
        pre_train_one_epoch(epoch, student, teacher, labeled_train_loader, optimizer,
                            device, rampup_length, max_lambda, start_ema_coef, end_ema_coef)
        val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(
            valid_loader, teacher, device,
            with_loss=True, with_standard_metrics=True, with_hd95=False)
        print(f"Epoch [{epoch+1}/{pre_epochs}] | loss: {val_loss:.4f} | dice: {val_dice:.4f} "
              f"| iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
        sys.stdout.flush()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher.state_dict(), best_model_path)
            print(f"  -> Best model updated (val_loss={val_loss:.4f})")

    pseudo_label_generator.load_state_dict(torch.load(best_model_path, map_location=device))

    # ================= Self-train =================
    print("\n--- Self-train ---")
    for epoch in range(epochs):
        self_train_one_epoch(epoch, student, teacher, pseudo_label_generator, train_loader,
                             optimizer, device, rampup_length, max_lambda, max_beta,
                             start_ema_coef, end_ema_coef)
        val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(
            valid_loader, teacher, device,
            with_loss=True, with_standard_metrics=True, with_hd95=False)
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {val_loss:.4f} | dice: {val_dice:.4f} "
              f"| iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
        sys.stdout.flush()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher.state_dict(), best_model_path)
            print(f"  -> Best model updated (val_loss={val_loss:.4f})")

    # ================= Test Evaluation =================
    print("\n--- Test Set Evaluation ---")
    teacher.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(
        test_loader, teacher, device,
        with_loss=True, with_standard_metrics=True, with_hd95=True)
    print(f"  Test Dice Coef: {test_dice:.4f}")
    print(f"  Test Jaccard Similarity: {test_jaccard:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test HD95: {test_hd95:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Proposed model with hyperparameters")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--dataset_name", type=str, default='OTU')
    parser.add_argument("--pre_epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_lambda", type=float, default=1.0)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--device_id", type=str, default='cuda:0')
    parser.add_argument("--best_model_path", type=str,
                        default=str(REPO_ROOT / "weight" / "proposed.pth"))
    parser.add_argument("--annotation_file", type=str, default=None)
    parser.add_argument("--start_ema_coef", type=float, default=0.99)
    parser.add_argument("--end_ema_coef", type=float, default=0.999)

    args = parser.parse_args()
    main(**vars(args))

import argparse
import numpy as np
import os
import cv2
import random
import copy
import torch
import torch.optim as optim
from ramp import sigmoid_rampup 
from model import Proposed
from data import get_dataloaders
from loss import *
import sys

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cv2.setRNGSeed(seed)
def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator()
g.manual_seed(seed)
#====================================================================================================================================

def get_ema_alpha(step, rampup_length, start, end):
    return start + (end - start) * sigmoid_rampup(current=step, rampup_length=rampup_length)

def update_ema_variables(model, ema_model, step, rampup_length, start, end, fixed_alpha=None):
    if fixed_alpha is not None:
        alpha = fixed_alpha
    else:
        alpha = get_ema_alpha(step, rampup_length, start, end)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# =================================================================================================================================================================

def pre_train_one_epoch(
                        epoch,
                        student,
                        teacher,
                        labeled_train_loader,
                        optimizer,
                        device,
                        rampup_length,
                        max_lambda,
                        start_ema_coef,
                        end_ema_coef):
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

def self_train_one_epoch(
                        epoch,
                        student,
                        teacher,
                        pseudo_label_generator,
                        train_loader,
                        optimizer,
                        device,
                        rampup_length,
                        max_lambda,
                        max_beta,
                        start_ema_coef,
                        end_ema_coef):
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
        
# =================================================================================================================================================================
# Evaluate
def evaluate(loader,
             teacher,
             device,
             with_loss,
             with_standard_metrics,
             with_hd95):

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

    return (
        running_loss / len(loader),
        running_dice / len(loader),
        running_jaccard / len(loader),
        running_recall / len(loader),
        running_precision / len(loader),
        running_hd95 / len(loader),
    )

# =================================================================================================================================================================

def main(
    image_size=256,
    batch_size=4,
    num_workers=0,
    pin_memory=False,
    labeled_ratio=0.1,
    dataset_name='OTU',
    pre_epochs=50,
    epochs=50,
    max_lambda=1.0,
    max_beta=1.0,
    start_ema_coef=0.99,
    end_ema_coef=0.999,
    learning_rate=0.001,
    device_id='cuda:0',
    best_model_path='WEIGHT/proposed.pth'
):

    device = torch.device(device_id)
    print("Device:", device)
    print("Proposed")

    student = Proposed().to(device)
    teacher = copy.deepcopy(student).to(device)
    pseudo_label_generator = copy.deepcopy(student).to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=0)

    labeled_train_loader, train_loader, valid_loader, test_loader = get_dataloaders(
        dataset_name, batch_size, num_workers, pin_memory, labeled_ratio
    )

    rampup_length = pre_epochs * len(labeled_train_loader) + epochs * len(train_loader)
    global global_step
    global_step = 0

    # ================= Pre-train =================
    print("\n--- Pre-train ---")
    best_loss = float('inf')
    for epoch in range(pre_epochs):
        pre_train_one_epoch(
                            epoch,
                            student,
                            teacher,
                            labeled_train_loader,
                            optimizer,
                            device,
                            rampup_length,
                            max_lambda,
                            start_ema_coef,
                            end_ema_coef
                        )
        val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(
                                                                                valid_loader,
                                                                                teacher,
                                                                                device,
                                                                                with_loss=True,
                                                                                with_standard_metrics=True,
                                                                                with_hd95=False
                                                                            )
        print(f"Epoch [{epoch+1}/{pre_epochs}] | loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
        sys.stdout.flush()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher.state_dict(), best_model_path)
            print(f"  → Best models updated (val_loss={val_loss:.4f})")

    pseudo_label_generator.load_state_dict(torch.load(best_model_path))

    # ================= Self-train =================
    print("\n--- Self-train ---")
    for epoch in range(epochs):
        self_train_one_epoch(
                            epoch,
                            student,
                            teacher,
                            pseudo_label_generator,
                            train_loader,
                            optimizer,
                            device,
                            rampup_length,
                            max_lambda,
                            max_beta,
                            start_ema_coef,
                            end_ema_coef
                        )
        val_loss, val_dice, val_jaccard, val_precision, val_recall, _ = evaluate(
                                                                                valid_loader,
                                                                                teacher,
                                                                                device,
                                                                                with_loss=True,
                                                                                with_standard_metrics=True,
                                                                                with_hd95=False
                                                                            )
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {val_loss:.4f} | dice: {val_dice:.4f} | iou: {val_jaccard:.4f} | precision: {val_precision:.4f} | recall: {val_recall:.4f}")
        sys.stdout.flush()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(teacher.state_dict(), best_model_path)
            print(f"  → Best models updated (val_loss={val_loss:.4f})")

    # ================= Test Evaluation =================
    print("\n--- Test Set Evaluation ---")
    teacher.load_state_dict(torch.load(best_model_path))
    _, test_dice, test_jaccard, test_recall, test_precision, test_hd95 = evaluate(
                                                                                test_loader,
                                                                                teacher,
                                                                                device,
                                                                                with_loss=True,
                                                                                with_standard_metrics=True,
                                                                                with_hd95=True
                                                                            )
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
    parser.add_argument("--best_model_path", type=str, default='weight/proposed2.pth')
    parser.add_argument("--start_ema_coef", type=float, default=0.99)
    parser.add_argument("--end_ema_coef", type=float, default=0.999)

    args = parser.parse_args()
    main(**vars(args))
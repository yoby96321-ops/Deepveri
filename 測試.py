from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

# =====================================================
# 資料集位置
# =====================================================
DATA_ROOT = Path(r"D:\欣\新增資料夾\real_vs_fake\real-vs-fake")
DIFF_ROOT = Path(r"D:\欣\新增資料夾\real_vs_fake\diffusion_faces_dataset")

TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR  = DATA_ROOT / "test"

DIFF_TRAIN_DIR = DIFF_ROOT / "train"
DIFF_VALID_DIR = DIFF_ROOT / "valid"
DIFF_TEST_DIR  = DIFF_ROOT / "test"

MODEL_DIR = Path(r"D:\欣\真假圖app\models")
MODEL_DIR.mkdir(exist_ok=True)

OLD_MODEL_PATH = MODEL_DIR / ""
MODEL_PATH = MODEL_DIR / "new_model.pth"

LOG_PATH = MODEL_DIR / "training_log_aug.csv"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix_aug.csv"

# =====================================================
# 訓練設定
# =====================================================
BATCH_SIZE = 32
EPOCHS = 3
LR = 0.00001
IMAGE_SIZE = 224
NUM_WORKERS = 0

NUM_SAMPLES_PER_EPOCH = 40000
DIFF_WEIGHT = 4.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("目前使用裝置：", device)

# =====================================================
# 檢查資料夾
# =====================================================
for folder in [
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    DIFF_TRAIN_DIR, DIFF_VALID_DIR, DIFF_TEST_DIR
]:
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

# =====================================================
# 圖片前處理
# =====================================================

# 訓練用：有資料增強
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),

    # 隨機裁切，但不要切太多，避免真圖特徵被破壞
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    # 模擬自拍左右鏡像
    transforms.RandomHorizontalFlip(p=0.5),

    # 模擬手機拍照小角度歪斜
    transforms.RandomRotation(5),

    # 模擬光線、對比、色彩差異
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
    ),

    # 模擬手機照片模糊、壓縮感
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 0.5)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 驗證/測試用：不要隨機增強，保持穩定評估
transform_eval = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# 資料集
# =====================================================
main_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
diff_train_dataset = datasets.ImageFolder(DIFF_TRAIN_DIR, transform=transform_train)

main_valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform_eval)
diff_valid_dataset = datasets.ImageFolder(DIFF_VALID_DIR, transform=transform_eval)

main_test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform_eval)
diff_test_dataset = datasets.ImageFolder(DIFF_TEST_DIR, transform=transform_eval)

train_dataset = ConcatDataset([main_train_dataset, diff_train_dataset])
valid_dataset = ConcatDataset([main_valid_dataset, diff_valid_dataset])
test_dataset  = ConcatDataset([main_test_dataset, diff_test_dataset])

print("類別對應：", main_train_dataset.class_to_idx)
print("Main Train 數量：", len(main_train_dataset))
print("Diff Train 數量：", len(diff_train_dataset))
print("Train 總數量：", len(train_dataset))
print("Valid 總數量：", len(valid_dataset))
print("Test 總數量：", len(test_dataset))

fake_idx = main_train_dataset.class_to_idx["fake"]
real_idx = main_train_dataset.class_to_idx["real"]

# =====================================================
# WeightedRandomSampler
# =====================================================
weights = [1.0] * len(main_train_dataset)

for i in range(len(diff_train_dataset)):
    weights.append(DIFF_WEIGHT)

print("weights 建立完成")

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=NUM_SAMPLES_PER_EPOCH,
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =====================================================
# 模型：預訓練 ResNet18
# =====================================================
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

if OLD_MODEL_PATH.exists():
    checkpoint = torch.load(OLD_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("已載入舊模型繼續微調：", OLD_MODEL_PATH)
else:
    print("沒有舊模型，使用 ImageNet 預訓練模型開始訓練")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# =====================================================
# 建立 CSV 紀錄檔
# =====================================================
with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Epoch",
        "Train Loss",
        "Train Acc",
        "Validation Loss",
        "Validation Acc",
        "Fake Precision",
        "Fake Recall",
        "Fake F1",
        "Real Precision",
        "Real Recall",
        "Real F1",
        "AUC",
        "TN",
        "FP",
        "FN",
        "TP"
    ])

print("CSV 紀錄檔建立完成：", LOG_PATH)

# =====================================================
# 評估函式
# =====================================================
def evaluate(loader, name="Valid"):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_real_probs = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=name):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            total_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_real_probs.extend(probs[:, real_idx].cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[fake_idx, real_idx],
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[fake_idx, real_idx]
    )

    try:
        auc = roc_auc_score(all_labels, all_real_probs)
    except:
        auc = 0

    return avg_loss, acc, precision, recall, f1, cm, auc

# =====================================================
# 開始訓練
# =====================================================
best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in loop:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    valid_loss, valid_acc, precision, recall, f1, cm, auc = evaluate(
        valid_loader,
        "Valid"
    )

    tn, fp, fn, tp = cm.ravel()

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss      : {avg_train_loss:.4f}")
    print(f"Train Acc       : {train_acc*100:.2f}%")
    print(f"Validation Loss : {valid_loss:.4f}")
    print(f"Validation Acc  : {valid_acc*100:.2f}%")

    print(f"Fake Precision  : {precision[0]:.4f}")
    print(f"Fake Recall     : {recall[0]:.4f}")
    print(f"Fake F1         : {f1[0]:.4f}")

    print(f"Real Precision  : {precision[1]:.4f}")
    print(f"Real Recall     : {recall[1]:.4f}")
    print(f"Real F1         : {f1[1]:.4f}")

    print(f"AUC             : {auc:.4f}")

    print("混淆矩陣：")
    print(cm)

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            avg_train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            precision[0],
            recall[0],
            f1[0],
            precision[1],
            recall[1],
            f1[1],
            auc,
            tn,
            fp,
            fn,
            tp
        ])

    if valid_acc > best_acc:
        best_acc = valid_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": main_train_dataset.class_to_idx,
            "best_acc": best_acc
        }, MODEL_PATH)

        print("已儲存最佳模型：", MODEL_PATH)

print("\n訓練完成")
print("最佳驗證準確率：", best_acc)

# =====================================================
# 測試
# =====================================================
if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("已載入最佳模型進行測試：", MODEL_PATH)

test_loss, test_acc, test_precision, test_recall, test_f1, test_cm, test_auc = evaluate(
    test_loader,
    "Test"
)

tn, fp, fn, tp = test_cm.ravel()

print("\n========== Test 測試結果 ==========")
print(f"Test Loss       : {test_loss:.4f}")
print(f"Test Acc        : {test_acc*100:.2f}%")

print(f"Fake Precision  : {test_precision[0]:.4f}")
print(f"Fake Recall     : {test_recall[0]:.4f}")
print(f"Fake F1         : {test_f1[0]:.4f}")

print(f"Real Precision  : {test_precision[1]:.4f}")
print(f"Real Recall     : {test_recall[1]:.4f}")
print(f"Real F1         : {test_f1[1]:.4f}")

print(f"AUC             : {test_auc:.4f}")

print("Test 混淆矩陣：")
print(test_cm)

with open(CONFUSION_MATRIX_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["", "Pred Fake", "Pred Real"])
    writer.writerow(["True Fake", tn, fp])
    writer.writerow(["True Real", fn, tp])

print("混淆矩陣 CSV 已儲存：", CONFUSION_MATRIX_PATH)
print("訓練紀錄 CSV 已儲存：", LOG_PATH)
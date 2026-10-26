from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

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

OLD_MODEL_PATH = MODEL_DIR / "best_model_better_diffusion.pth"
MODEL_PATH = MODEL_DIR / "best_model_better_diffusion.pth"

# =====================================================
# 訓練設定
# =====================================================
BATCH_SIZE = 32
EPOCHS = 6
LR = 0.00005
IMAGE_SIZE = 224
NUM_WORKERS = 0

NUM_SAMPLES_PER_EPOCH = 60000
DIFF_WEIGHT = 10.0

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
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

# 如果有舊模型，就載入舊模型繼續微調
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
# 評估函式
# =====================================================
def evaluate(loader, name="Valid"):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=name):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    avg_loss = total_loss / len(loader)
    acc = correct / total

    fake_idx = main_train_dataset.class_to_idx["fake"]
    real_idx = main_train_dataset.class_to_idx["real"]

    fake_acc = class_correct[fake_idx] / class_total[fake_idx] if class_total[fake_idx] > 0 else 0
    real_acc = class_correct[real_idx] / class_total[real_idx] if class_total[real_idx] > 0 else 0

    return avg_loss, acc, fake_acc, real_acc

# =====================================================
# 開始訓練
# =====================================================
best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

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

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    valid_loss, valid_acc, valid_fake_acc, valid_real_acc = evaluate(valid_loader, "Valid")
    avg_train_loss = train_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Valid Acc : {valid_acc*100:.2f}%")
    print(f"Fake Acc  : {valid_fake_acc*100:.2f}%")
    print(f"Real Acc  : {valid_real_acc*100:.2f}%")

    if valid_acc > best_acc:
        best_acc = valid_acc

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": main_train_dataset.class_to_idx,
            "best_acc": best_acc
        }, MODEL_PATH)

        print("已儲存最佳模型：", MODEL_PATH)

print("訓練完成")
print("最佳驗證準確率：", best_acc)

# =====================================================
# 測試
# =====================================================
if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc, test_fake_acc, test_real_acc = evaluate(test_loader, "Test")

print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc*100:.2f}%")
print(f"Fake Acc  : {test_fake_acc*100:.2f}%")
print(f"Real Acc  : {test_real_acc*100:.2f}%")
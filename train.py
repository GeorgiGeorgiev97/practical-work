import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from mheads import MHeads
import mlflow

class OCTDataset(Dataset):
    """Dataset for OCT"""

    # Class mapping for OCT dataset
    CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __init__(self, image_dir, max_images=None, transform=None, return_labels=True, include_classes=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.return_labels = return_labels

        image_dir = Path(image_dir)

        # Use a subset of classes if specified, remapping indices to 0..N-1
        active_classes = include_classes if include_classes is not None else self.CLASS_NAMES
        active_class_to_idx = {name: idx for idx, name in enumerate(active_classes)}

        subdirs = [d for d in image_dir.iterdir() if d.is_dir()]

        if subdirs and any(d.name in self.CLASS_NAMES for d in subdirs):
            # Load from each class subfolder
            for class_name in active_classes:
                class_dir = image_dir / class_name
                if class_dir.exists():
                    class_images = list(class_dir.glob('*.jpeg'))
                    self.image_paths.extend(class_images)
                    self.labels.extend([active_class_to_idx[class_name]] * len(class_images))

        else:
            # Single folder: all images get label 0
            self.image_paths = list(image_dir.rglob('*.jpeg'))
            self.labels = [0] * len(self.image_paths)

        # Limit images if needed
        if max_images and len(self.image_paths) > max_images:
            indices = np.random.choice(len(self.image_paths), max_images, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def train_epoch(model, dataloader, epoch, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        model._optimizer.zero_grad()
        output = model._model_instance(images)
        loss = model.calculate_loss(output, labels)

        loss.backward()
        model._optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    """Validate and return loss, accuracy, F1"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model._model_instance(images)
            loss = model.calculate_loss(output, labels)
            total_loss += loss.item()

            # average heads
            probs = torch.nn.functional.softmax(output, dim=1)
            mean_probs = torch.mean(probs, dim=-1)  # Average across heads
            preds = torch.argmax(mean_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss, val_acc, val_f1


def evaluate_with_entropy(model, dataset, device, desc="Evaluating"):
    """Evaluate and calculate entropy"""
    model.eval()
    entropies = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            output = model._model_instance(image)

            probs = torch.nn.functional.softmax(output, 1)
            probs = torch.mean(probs, dim=-1).cpu().numpy()[0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

    return np.array(entropies)


def main():

    # Config
    NUM_TRAIN_IMAGES = None     # None to use all images
    NUM_TEST_INLIER = 500        # Number of inliers to test
    NUM_TEST_OUTLIER = None      # None to test all outliers
    NUM_EPOCHS = 20              # Number of epochs
    BATCH_SIZE = 16              # Batch size
    IMAGE_SIZE = 224             # Image size 224/128

    # 3-classes in-distribution: CNV, DRUSEN, NORMAL
    TRAIN_CLASSES = ['CNV', 'DRUSEN', 'NORMAL']

    # Paths
    oct_test_dir = "data/OCT/test"
    oct_train_dir = "data/OCT/train"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    mlflow.set_experiment("oct-ood-detection")
    mlflow.start_run(run_name=f"densenet-3class-dme-ood-{NUM_EPOCHS}ep-{IMAGE_SIZE}px")

    config = {
        'input_shape': [3, IMAGE_SIZE, IMAGE_SIZE],
        'num_classes': 3,  # CNV, DRUSEN, NORMAL
        'backbone': 'densenet',
        'optimizer_type': 'adam',
        'optimizer_mom': 0.9,
        'optimizer_wd': 0.0001,
        'head_depth': 16,
        'num_heads': 10,
        'mhead_random': 0.02,
        'mhead_eps': 0.05,
        'learning_rate': 0.0001,
        'dropout_rate': 0.1,
        'layers_per_block': [4, 4, 4],
        'growth_rate': 32,
        'bottleneck': True,
        'reduction': 0.5,
        'pool_initial': True,
        'pool_final': False,
        'valid_padding': False,
        'bottleneck_rate': 2,
        'multi_gpu': False
    }

    model = MHeads(name='oct-trained', description='Trained on OCT for OOD detection')
    model.configure(**config)
    model.build()

    # Log to mlflow
    mlflow.log_params({
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "num_train_images": NUM_TRAIN_IMAGES,
        "learning_rate": config['learning_rate'],
        "num_heads": config['num_heads'],
        "head_depth": config['head_depth'],
        "backbone": config['backbone'],
        "dropout_rate": config['dropout_rate'],
        "growth_rate": config['growth_rate'],
        "mhead_random": config['mhead_random'],
        "mhead_eps": config['mhead_eps'],
    })


    print(f"Training images: {NUM_TRAIN_IMAGES}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = OCTDataset(oct_train_dir, max_images=NUM_TRAIN_IMAGES, transform=transform,
                              include_classes=TRAIN_CLASSES)

    # Split into train/val (80/20)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=len(TRAIN_CLASSES))
    print(f"Class distribution: " + ", ".join(f"{c}={class_counts[i]}" for i, c in enumerate(TRAIN_CLASSES)))

    # Create weighted sampler (inverse frequency)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, epoch, device)
        train_losses.append(train_loss)

        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        gap = train_loss - val_loss
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | gap={gap:+.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "train_val_gap": gap
        }, step=epoch)

    print(f"\nHead usage: {dict(model.head_report['training'])}")

    model.save("oct_trained_model.pth")
    print("\nModel saved to: oct_trained_model.pth")

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Inliers: CNV, DRUSEN, NORMAL from OCT test set
    inlier_dataset = OCTDataset(oct_test_dir, max_images=NUM_TEST_INLIER, transform=test_transform,
                                include_classes=TRAIN_CLASSES)
    inlier_entropies = evaluate_with_entropy(model, inlier_dataset, device, "Inliers (OCT: CNV/DRUSEN/NORMAL)")

    # OOD: DME from OCT test set
    dme_test_dir = Path(oct_test_dir) / 'DME'
    dme_dataset = OCTDataset(dme_test_dir, max_images=NUM_TEST_OUTLIER, transform=test_transform)
    outlier_entropies = evaluate_with_entropy(model, dme_dataset, device, "OOD: DME (OCT test)")
    outlier_labels = ['DME'] * len(outlier_entropies)

    print(f"INLIERS  (CNV/DRUSEN/NORMAL): {np.mean(inlier_entropies):.4f} ± {np.std(inlier_entropies):.4f}")
    print(f"OUTLIERS (DME):               {np.mean(outlier_entropies):.4f} ± {np.std(outlier_entropies):.4f}")
    print(f"Difference: {np.mean(outlier_entropies) - np.mean(inlier_entropies):.4f}")

    y_true = np.concatenate([np.ones(len(inlier_entropies)), np.zeros(len(outlier_entropies))])
    y_scores = np.concatenate([inlier_entropies, outlier_entropies])

    # Flipped logic
    y_scores = -y_scores
    auroc = roc_auc_score(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # threshold at 95% TPR
    idx_95 = np.argmax(tpr >= 0.95)
    fpr_at_95tpr = fpr[idx_95]

    # best threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred = (y_scores >= best_threshold).astype(int)
    threshold_acc = accuracy_score(y_true, y_pred)
    threshold_precision = precision_score(y_true, y_pred, zero_division=0)
    threshold_recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"\nAUROC for OOD detection: {auroc:.4f}")
    print(f"FPR @ 95% TPR: {fpr_at_95tpr:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Accuracy: {threshold_acc:.4f}")
    print(f"Precision: {threshold_precision:.4f}")
    print(f"Recall: {threshold_recall:.4f}")

    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, marker='s', label='Val Loss', color='orange')
    plt.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss (gap = overfitting)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accs, marker='o', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy (final: {val_accs[-1]:.4f})')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1s, marker='o', color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (macro)')
    plt.title(f'Validation F1 Macro (final: {val_f1s[-1]:.4f})')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(inlier_entropies, bins=30, alpha=0.6, label='Inliers', color='blue', density=True)
    plt.hist(outlier_entropies, bins=30, alpha=0.6, label='Outliers', color='red', density=True)
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title(f'Entropy Distributions (AUROC={auroc:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("Saved: training_results.png")

    plt.figure(figsize=(12, 6))

    plt.boxplot([inlier_entropies, outlier_entropies],
                labels=['CNV/DRUSEN/NORMAL (In)', 'DME (OOD)'],
                patch_artist=True)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Entropy')
    plt.title('Entropy by Category')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('per_disease_entropy.png', dpi=150, bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    # Best threshold point
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5,
                label=f'Best threshold (TPR={tpr[best_idx]:.2f}, FPR={fpr[best_idx]:.2f})')
    plt.scatter([fpr_at_95tpr], [0.95], color='orange', s=100, zorder=5, marker='s',
                label=f'95% TPR (FPR={fpr_at_95tpr:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for OOD Detection')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')

    mlflow.log_metrics({
        "final_val_acc": val_accs[-1],
        "final_val_f1": val_f1s[-1],
        "final_val_loss": val_losses[-1],
        "final_train_loss": train_losses[-1],
        "auroc": auroc,
        "fpr_at_95tpr": fpr_at_95tpr,
        "best_threshold": best_threshold,
        "threshold_accuracy": threshold_acc,
        "threshold_precision": threshold_precision,
        "threshold_recall": threshold_recall,
        "inlier_entropy_mean": float(np.mean(inlier_entropies)),
        "outlier_entropy_mean": float(np.mean(outlier_entropies)),
        "entropy_difference": float(np.mean(outlier_entropies) - np.mean(inlier_entropies)),
        "num_inliers": len(inlier_entropies),
        "num_outliers": len(outlier_entropies),
    })
    for head_id, count in model.head_report['training'].items():
        mlflow.log_metric(f"head_{head_id}_usage", count)

    mlflow.log_artifact("training_results.png")
    mlflow.log_artifact("per_disease_entropy.png")
    mlflow.log_artifact("roc_curve.png")
    mlflow.log_artifact("oct_trained_model.pth")

    mlflow.end_run()

if __name__ == "__main__":
    main()

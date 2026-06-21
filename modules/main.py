# main.py
import argparse
import os

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18+LSTM video action classifier."
    )
    parser.add_argument("--train-data", default=config.TRAIN_DATA_PATH)
    parser.add_argument("--test-data", default=config.TEST_DATA_PATH)
    parser.add_argument("--labels", default=config.LABEL_CSV)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from data_loader import SimpleAVIDataset
    from embedding_analysis import run_embedding_analysis
    from model import SimpleCNNLSTM
    from train import train_epoch, validate
    from utils import ensure_dir, plot_training_history, save_json, save_predictions, set_seed

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    print("=" * 60)
    print("AVI Video Classification Training")
    print("=" * 60)

    print("Checking paths...")
    print(f"Training data path exists: {os.path.exists(args.train_data)}")
    print(f"Test data path exists: {os.path.exists(args.test_data)}")
    print(f"Label file exists: {os.path.exists(args.labels)}")

    print("\nLoading label file...")
    label_df = pd.read_csv(args.labels, header=None, names=['video_name', 'label_name', 'label_id'])
    print(f"Label file shape: {label_df.shape}")
    print(f"First 5 rows:\n{label_df.head()}")

    all_classes = sorted(label_df['label_name'].unique().tolist())
    num_classes = len(all_classes)
    print(f"\nTotal {num_classes} classes")

    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"\nUsing device: {config.DEVICE}")
    print(f"Writing outputs to: {output_dir}")

    print("\nCreating training dataset...")
    train_dataset = SimpleAVIDataset(
        video_dir=args.train_data,
        label_df=label_df,
        num_frames=args.num_frames,
        img_size=args.img_size,
        is_train=True,
        class_to_idx=class_to_idx
    )

    if len(train_dataset) < 2:
        print("Error: need at least 2 labeled samples for train/validation split.")
        return

    val_size = max(1, int(config.VAL_SPLIT * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    if train_size < 1:
        print("Error: training split is empty after validation split.")
        return

    split_generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=split_generator
    )

    print("Creating test dataset...")
    test_dataset = SimpleAVIDataset(
        video_dir=args.test_data,
        label_df=None,
        num_frames=args.num_frames,
        img_size=args.img_size,
        is_train=False,
        class_to_idx=class_to_idx
    )

    print(f"Training set: {len(train_subset)} samples")
    print(f"Validation set: {len(val_subset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"\nCreating model (number of classes: {num_classes})...")
    model = SimpleCNNLSTM(
        num_classes=num_classes,
        hidden_dim=config.HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS
    ).to(config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nStarting training, total {args.epochs} epochs...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.2f}%")
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, os.path.join(output_dir, "best_simple_model.pth"))
            print(f"Saved best model. Validation accuracy: {val_acc:.2f}%")

    print("\nRunning validation embedding/distribution analysis...")
    validation_metrics = run_embedding_analysis(
        model,
        val_loader,
        config.DEVICE,
        idx_to_class,
        output_dir,
        split_name="val",
    )

    print("\nMaking predictions on test set...")
    model.eval()
    predictions = []
    video_files = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, _ = test_dataset[i]
            x = x.unsqueeze(0).to(config.DEVICE)
            outputs = model(x)
            _, predicted = outputs.max(1)

            predictions.append(predicted.item())
            video_files.append(os.path.basename(test_dataset.video_files[i]))

            if i % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")

    save_predictions(
        predictions,
        video_files,
        idx_to_class,
        os.path.join(output_dir, 'test_predictions.csv')
    )
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    save_json({
        "config": {
            "train_data": args.train_data,
            "test_data": args.test_data,
            "labels": args.labels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_frames": args.num_frames,
            "img_size": args.img_size,
            "seed": args.seed,
            "device": str(config.DEVICE),
            "num_classes": num_classes,
        },
        "history": history,
        "best_val_acc_percent": best_val_acc,
        "validation_metrics": validation_metrics,
    }, os.path.join(output_dir, "metrics.json"))

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved to: {os.path.join(output_dir, 'final_model.pth')}")


if __name__ == "__main__":
    main()

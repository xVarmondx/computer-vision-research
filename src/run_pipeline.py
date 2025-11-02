import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from src.config import *
from src.data_loader import load_dataframe, get_dataloaders, get_subset_loader
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_subset, plot_metrics, plot_confusion_matrix


def main(force_train):
    """
    Main function to run the entire pipeline (training and evaluation).
    """

    print(f"--- Running pipeline on device: {DEVICE} ---")

    print("\n[STEP 1/5] Loading metadata...")
    data_df = load_dataframe(DATA_PATH)
    if data_df.empty:
        print(f"ERROR: No data found in {DATA_PATH}. Aborting.")
        return
    print(f"Found {len(data_df)} files.")

    print("\n[STEP 2/5] Splitting data and creating DataLoaders...")
    train_loader, val_loader, test_loader, test_df = get_dataloaders(
        data_df, GLOBAL_CLASS_TO_IDX, BATCH_SIZE
    )
    print("DataLoaders (train/val/test) successfully created.")

    print("\n[STEP 3/5] Initializing ResNet-18 model...")
    model = create_model(NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print("Model, loss criterion, and optimizer defined.")

    print("\n[STEP 4/5] Training model...")
    history = None

    if force_train or not os.path.exists(MODEL_PATH):
        if force_train:
            print("Force-train option enabled. Starting model training...")
        else:
            print(f"Weights file '{MODEL_PATH}' not found. Starting new training run...")

        model, history = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, scheduler, DEVICE, NUM_EPOCHS, MODEL_PATH
        )

        if history:
            plot_metrics(history)
    else:
        print(f"Found trained model at '{MODEL_PATH}'. Skipping training.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("\n[STEP 5/5] Evaluating on test set...")

    idx_to_class = {v: k for k, v in GLOBAL_CLASS_TO_IDX.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    print("\n--- CLASSIFICATION REPORT: TEST SET (OVERALL) ---")
    all_labels_test, all_pred_test, test_report, target_names = evaluate_subset(
        test_loader, model, DEVICE, NUM_CLASSES,
        GLOBAL_CLASS_TO_IDX, model_path_to_load=None
    )
    print(test_report)

    plot_confusion_matrix(all_labels_test, all_pred_test, target_names)

    print("\n--- Subset Analysis (Hand-Drawn vs Stamp) ---")

    test_hand_drawn_df = test_df[test_df['image_type'].str.contains('Hand-Drawn')].reset_index(drop=True)
    test_stamp_df = test_df[test_df['image_type'].str.contains('Stamp')].reset_index(drop=True)

    val_test_transforms = transforms.Compose([
        transforms.Resize((130, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader_hand_drawn = get_subset_loader(test_hand_drawn_df, val_test_transforms, GLOBAL_CLASS_TO_IDX, BATCH_SIZE)
    loader_stamp = get_subset_loader(test_stamp_df, val_test_transforms, GLOBAL_CLASS_TO_IDX, BATCH_SIZE)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT: HAND-DRAWN")
    print("=" * 60)
    _, _, report_hd, _ = evaluate_subset(loader_hand_drawn, model, DEVICE, NUM_CLASSES, GLOBAL_CLASS_TO_IDX)
    print(report_hd)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT: STAMP / DIGITAL GRAPHICS")
    print("=" * 60)
    _, _, report_st, _ = evaluate_subset(loader_stamp, model, DEVICE, NUM_CLASSES, GLOBAL_CLASS_TO_IDX)
    print(report_st)

    print("\n--- Pipeline finished successfully! ---")


if __name__ == "__main__":
    """
    Entry point for the script, handling command line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs the full model training and evaluation pipeline.")

    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Forces model retraining, even if the weights file already exists."
    )

    args = parser.parse_args()

    main(force_train=args.force_train)
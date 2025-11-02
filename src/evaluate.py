import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def evaluate_subset(loader, model, device, num_classes, class_to_idx, model_path_to_load=None):
    """
    It evaluates the model on the given loader and returns the labels and predictions.
    This function is flexible: it either uses an already loaded model or loads the weights from a file
    (if model_path_to_load is provided)
    """

    if not loader or len(loader.dataset) == 0:
        return [], [], []

    if model_path_to_load:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        try:
            model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku wag modelu: {model_path_to_load}.")
            return [], [], []

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    all_class_indices = sorted(class_to_idx.values())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        labels=all_class_indices,
        zero_division=0
    )

    return all_labels, all_preds, report, target_names


def plot_metrics(history):
    """
    Plots training and validation loss and accuracy over epochs.
    Requires 'history' dictionary generated during training.
    """

    train_acc = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in history['train_acc']]
    val_acc = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in history['val_acc']]
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, preds, class_names):
    """
    Plots the Confusion Matrix as a heatmap.
    """
    if not labels or not preds:
        print("No data to generate Confusion Matrix.")
        return

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='g',  # 'g' wyłącza notację naukową
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Model Prediction', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.show()


def visualize_model_predictions(model, loader, device, class_to_idx, num_images=10):
    """
    Visualizes image predictions, showing True label, Predicted label, and Confidence.
    Arranged in 4 columns per row.
    """
    model.eval()
    model.to(device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    images_so_far = 0
    rows = int(np.ceil(num_images / 4))
    fig = plt.figure(figsize=(18, rows * 4))

    with torch.no_grad():
        try:
            inputs, labels = next(iter(loader))
        except StopIteration:
            print("Error: Loader is empty.")
            return

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probs.topk(1, dim=1)

        preds = top_class.view(-1)

        for j in range(inputs.size()[0]):
            if images_so_far >= num_images:
                break

            images_so_far += 1
            ax = plt.subplot(rows, 4, images_so_far)
            ax.axis('off')

            pred_label = idx_to_class[preds[j].item()]
            true_label = idx_to_class[labels[j].item()]
            pred_prob = top_p[j].item() * 100

            color = "green" if pred_label == true_label else "red"

            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConfidence: {pred_prob:.1f}%',
                         color=color, fontsize=12)

            img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)

    plt.tight_layout(h_pad=2.0)
    plt.show()


def plot_grad_cam(model, target_layer, loader, device, class_to_idx, img_index=0, target_type='true'):
    """
    Plots the Grad-CAM heatmap for a selected image, showing where the model focuses.
    Target type can be 'true' (actual class) or 'pred' (predicted class, useful for error analysis).
    """

    model.eval()
    model.to(device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    try:
        inputs, labels = next(iter(loader))
        input_tensor = inputs[img_index].unsqueeze(0).to(device)
        true_label_idx = labels[img_index].item()
    except StopIteration:
        print("Error: Loader is empty.")
        return
    except IndexError:
        print(f"Error: Index {img_index} is out of batch range. Try a smaller index.")
        return


    target_layers = [target_layer]
    cam = GradCAM(model=model, target_layers=target_layers)

    output = model(input_tensor)
    _, pred_idx = torch.max(output, 1)

    if target_type == 'pred':
        target_idx = pred_idx.item()
        target_name = idx_to_class[target_idx]
    else:
        target_idx = true_label_idx
        target_name = idx_to_class[target_idx]

    targets = [ClassifierOutputTarget(target_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    img = input_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = std * img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    output = model(input_tensor)
    _, pred_idx = torch.max(output, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Orginal\nTrue: {idx_to_class[true_label_idx]}")
    axes[0].axis('off')

    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title("Heatmapa Grad-CAM")
    axes[1].axis('off')

    axes[2].imshow(visualization)
    axes[2].set_title(f"Overlay Heatmap\nCAM Target: {target_name}\nPrediction: {idx_to_class[pred_idx.item()]}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

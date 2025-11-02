import torch

DATA_PATH = "../dataset"
MODEL_PATH = "best_model_weights.pth"

BATCH_SIZE = 32
NUM_CLASSES = 10

GLOBAL_CLASS_TO_IDX = {
    'anchor': 0, 'balloon': 1, 'bicycle': 2, 'envelope': 3,
    'paper_boat': 4, 'peace_symbol': 5, 'smiley': 6,
    'speech_bubble': 7, 'spiral': 8, 'thumb': 9
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
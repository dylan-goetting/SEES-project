import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms import ToTensor

# Paths
PROCESSED_DATA_PATH = "data/processed/filtered_vqa_with_links.json"

# Load processed VQA data
try:
    with open(PROCESSED_DATA_PATH, "r") as f:
        dataset = json.load(f)
    print(f"✅ Successfully loaded dataset from {PROCESSED_DATA_PATH}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {PROCESSED_DATA_PATH}")
    exit(1)  # Stop execution if file is missing

# Inspect dataset structure
print("\n🔍 First 3 dataset entries:")
for i in range(min(3, len(dataset))):
    print(json.dumps(dataset[i], indent=4))

# Dummy inference function (Replace with real model inference)
def inference(question, image):
    """Simulate inference returning an answer and a 24x24 matrix."""
    answer = "dummy_answer"  # Replace with actual model output
    matrix = np.random.rand(24, 24)  # Random 24x24 matrix for demonstration
    return answer, matrix

# Define PyTorch Dataset class
class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.to_tensor = ToTensor()  # Convert images to tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract relevant fields
        question_text = sample.get("question_text", "[MISSING QUESTION TEXT]")
        image_url = sample.get("image_url", None)
        category = sample.get("category", "[MISSING CATEGORY]")
        answer = sample.get("answer", "[MISSING ANSWER]")

        # Load image from URL if available
        image = None
        if image_url:
            try:
                response = requests.get(image_url, timeout=5)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = self.to_tensor(image)  # Convert to tensor
            except Exception as e:
                print(f"❌ Error loading image from {image_url}: {e}")
                image = torch.zeros(3, 224, 224)  # Default black image

        return question_text, image, category, answer

# Load dataset into DataLoader
batch_size = 16  # Adjust based on available GPU/CPU memory
vqa_dataset = VQADataset(dataset)
dataloader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True)

# Entropy function to compute the entropy of a probability distribution
def compute_entropy(probabilities):
    """Calculate entropy of a probability distribution."""
    probabilities = np.array(probabilities)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)

# Iterate through DataLoader and process each batch
for batch in dataloader:
    questions, images, categories, answers = batch

    # Run inference on the batch
    answers_out = []
    matrices = []
    for q, img in zip(questions, images):
        ans, matrix = inference(q, img)
        answers_out.append(ans)
        matrices.append(matrix)

    # Convert answers into a dummy probability distribution
    answer_probs = np.random.dirichlet(np.ones(len(answers_out)), size=1)[0]

    # Compute entropy
    entropy = compute_entropy(answer_probs)

    # Log batch processing results
    print(f"✅ Batch Processed | Entropy: {entropy:.4f} | Categories: {set(categories)}")

print(len(vqa_dataset))

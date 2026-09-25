import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1


class FaceEmbeddingModel:

    def __init__(self):
        # Use CPU
        self.device = torch.device("cpu")

        # Load pretrained FaceNet model
        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

    def generate_embedding(self, face_image):
        """
        Convert a preprocessed face image into
        a 512-dimensional face embedding.
        """

        # face_image is currently:
        # (160, 160, 3), RGB, float32, values 0-1

        # Convert NumPy → PyTorch tensor
        tensor = torch.from_numpy(face_image)

        # Change:
        # (H, W, C) → (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        # Add batch dimension:
        # (3, 160, 160) → (1, 3, 160, 160)
        tensor = tensor.unsqueeze(0)

        # Move to CPU
        tensor = tensor.to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model(tensor)

        # Convert PyTorch tensor → NumPy
        embedding = embedding.squeeze(0).numpy()

        return embedding
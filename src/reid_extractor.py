import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class ReIDExtractor:
    """
    Lightweight Deep Learning Feature Extractor for Re-Identification.
    Uses a headless MobileNetV2 to generate a 1280-dimensional appearance signature
    from a cropped person bounding box.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load MobileNetV2 with standard ImageNet weights
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classification head to output the raw 1280-d features
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        # Standard ReID transforms (Resize to 256x128)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, boxes: list) -> list:
        """
        Extracts L2-normalized embeddings for a list of bounding boxes.
        Args:
            frame: Original video frame
            boxes: List of bounding boxes in [x1, y1, x2, y2] format
        Returns:
            List of 1280-d embeddings (as pure python lists for JSON serialization)
        """
        if boxes is None or len(boxes) == 0:
            return []

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            
            # Fallback for out-of-bounds or zero-area crops
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            
            # Convert BGR (OpenCV) to RGB (Torchvision)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            tensor = self.transform(crop_rgb)
            crops.append(tensor)
        
        batch = torch.stack(crops).to(self.device)
        embeddings = self.model(batch)
        
        # L2 normalize embeddings so we can use Cosine Similarity cleanly
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()

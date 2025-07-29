import os
import sys
import glob
import numpy as np
import cv2
import torch

from repvit_sam import sam_model_registry, SamPredictor
from repvit_sam.utils.transforms import ResizeLongestSide


class RepVITSamWrapper:
    def __init__(self, ckpt_filepath, model_type="repvit", 
                 device=None, multimaskoutput=False):
        if not os.path.exists(ckpt_filepath):
            raise Exception("SAM checkpoint path not valid")
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        self.sam_model = sam_model_registry[model_type](checkpoint=ckpt_filepath)
        self.sam_model.to(device=device)
        self.sam_model.eval()

        self.sam_predictor = SamPredictor(self.sam_model)

        self.multimaskoutput = multimaskoutput

    def set_image(self, image):
        self.sam_predictor.set_image(image)

    def get_image_embeddings(self):
        embeddings = self.sam_predictor.features.squeeze().cpu().numpy().astype(np.float32)
        return embeddings

    def predict(self, input_points, input_labels):
        assert type(input_points) == np.ndarray
        assert type(input_labels) == np.ndarray
        assert input_points.shape[0] == input_labels.shape[0]

        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            input_points, input_labels,
            multimask_output=self.multimaskoutput,
            return_logits=True
        )

        return masks, ious, low_res_mask_logits
import re
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #
class BaseFeatureExtractor(nn.Module, ABC):
    """Abstract base class for image‑ and text‑embedding extractors."""

    @abstractmethod
    def extract_image_features(self, images) -> torch.Tensor: ...
    @abstractmethod
    def extract_text_features(self, texts: List[str]) -> torch.Tensor: ...

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @property
    @abstractmethod
    def input_resolution(self) -> int: ...


# --------------------------------------------------------------------------- #
# OpenAI CLIP (Hugging Face)
# --------------------------------------------------------------------------- #
class OpenAICLIPFeatureExtractor(BaseFeatureExtractor):
    MODEL_HIDDEN_SIZES = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-large-patch14-336": 768,
    }

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.model_name, self.device = model_name, device

        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)          # default normalisation
        self.model     = CLIPModel.from_pretrained(model_name).to(device).eval()
        self._feature_dim = self.MODEL_HIDDEN_SIZES.get(model_name, 768)
        
        # Disable gradient computation for all parameters (performance)
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    def extract_image_features(self, images) -> torch.Tensor:
        """`images` = list[PIL]  (processor handles resize/crop/normalise)."""
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            inputs   = self.processor(images=images, return_tensors="pt")
            # Move to device with non_blocking for overlapped data transfer
            inputs   = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            embeds   = self.model.get_image_features(**inputs)
            return F.normalize(embeds, dim=-1)

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True)
            # Move to device with non_blocking for overlapped data transfer
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            embeds = self.model.get_text_features(**inputs)
            return F.normalize(embeds, dim=-1)

    # ------------------------------------------------------------------ #
    @property
    def feature_dim(self) -> int: return self._feature_dim

    @property
    def input_resolution(self) -> int: return 336 if "336" in self.model_name else 224


# --------------------------------------------------------------------------- #
# OpenCLIP
# --------------------------------------------------------------------------- #
class OpenCLIPFeatureExtractor(BaseFeatureExtractor):
    MODEL_HIDDEN_SIZES = {
        "vit-b-32": 512, "vit-b-16": 512,
        "vit-l-14": 768, "vit-h-14": 1024,
    }

    def __init__(self, model_name="ViT-B-32", device="cuda"):
        super().__init__()
        self.model_name, self.device = model_name, device

        import open_clip
        self.model, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion400m_e32"
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._feature_dim = self.MODEL_HIDDEN_SIZES.get(model_name.lower(), 512)
        
        # Disable gradient computation for all parameters (performance)
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------ #
    def _preprocess_batch(self, images) -> torch.Tensor:
        """Apply OpenCLIP's transform & stack to a GPU tensor."""
        tensors = torch.stack([self.preprocess(img) for img in images])
        return tensors.to(self.device, non_blocking=True)

    def extract_image_features(self, images) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            embeds = self.model.encode_image(self._preprocess_batch(images))
            return F.normalize(embeds, dim=-1)

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            tokens = self.tokenizer(texts).to(self.device, non_blocking=True)
            embeds = self.model.encode_text(tokens)
            return F.normalize(embeds, dim=-1)

    # ------------------------------------------------------------------ #
    @property
    def feature_dim(self) -> int: return self._feature_dim
    @property
    def input_resolution(self) -> int: return 224


# --------------------------------------------------------------------------- #
# Base SigLIP (v1) & SigLIP 2  (Hugging Face, google/*)
# --------------------------------------------------------------------------- #
def _parse_siglip_resolution(model_name: str, default: int = 224) -> int:
    m = re.search(r"patch\d+-(\d+)", model_name)
    return int(m.group(1)) if m else default


class _SigLIPBase(BaseFeatureExtractor):
    def _init_common(self, model_name, device):
        from transformers import AutoProcessor, AutoModel
        self.model_name, self.device = model_name, device
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.model     = AutoModel.from_pretrained(model_name, trust_remote_code=True) \
                                     .to(device).eval()
        self._feature_dim = self.MODEL_HIDDEN_SIZES.get(model_name, 768)
        
        # Disable gradient computation for all parameters (performance)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Cache resolution parsing (regex is expensive, avoid repeated calls)
        self._cached_resolution = _parse_siglip_resolution(model_name)
        
        # Cache max text length to ensure truncation works
        # SigLIP models have varying max lengths (64 for some, 256 for others)
        if hasattr(self.model.config, 'text_config'):
            self._max_text_length = self.model.config.text_config.max_position_embeddings
        else:
            print(f"Warning: No text config found for {model_name}, using default max length 64")
            self._max_text_length = 64

    # same implementation for image / text across SigLIP flavours
    def extract_image_features(self, images) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            inputs  = self.processor(images=images, return_tensors="pt")
            # Move to device with non_blocking for overlapped data transfer
            inputs  = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            embeds  = self.model.get_image_features(**inputs)
            return F.normalize(embeds, dim=-1)

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True, 
                                    max_length=self._max_text_length)
            # Move to device with non_blocking for overlapped data transfer
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            embeds = self.model.get_text_features(**inputs)
            return F.normalize(embeds, dim=-1)

    @property
    def feature_dim(self) -> int: return self._feature_dim
    @property
    def input_resolution(self) -> int: return self._cached_resolution


class SigLIPFeatureExtractor(_SigLIPBase):
    MODEL_HIDDEN_SIZES = {
        "google/siglip-base-patch16-224": 768,
        "google/siglip-large-patch16-384": 1024,
        "google/siglip-so400m-patch14-224": 1152,
        "google/siglip-so400m-patch14-384": 1152,
        "google/siglip-so400m-patch16-256": 1152,
        "google/siglip-so400m-patch16-384": 1152,
        "google/siglip-so400m-patch16-512": 1152,
        "google/siglip-so400m-patch16-naflex": 1152,
        "google/siglip-giant-opt-patch16-256": 1280,
        "google/siglip-giant-opt-patch16-384": 1280,
    }

    def __init__(self, model_name="google/siglip-base-patch16-224", device="cuda"):
        super().__init__()
        self._init_common(model_name, device)


class SigLIP2FeatureExtractor(_SigLIPBase):
    MODEL_HIDDEN_SIZES = {
        "google/siglip2-base-patch16-224": 768,
        "google/siglip2-base-patch16-256": 768,
        "google/siglip2-base-patch16-384": 768,
        "google/siglip2-base-patch16-512": 768,
        "google/siglip2-base-patch16-naflex": 768,
        "google/siglip2-base-patch32-256": 768,
        "google/siglip2-large-patch16-256": 1024,
        "google/siglip2-large-patch16-384": 1024,
        "google/siglip2-large-patch16-512": 1024,
        "google/siglip2-so400m-patch14-224": 1152,
        "google/siglip2-so400m-patch14-384": 1152,
        "google/siglip2-so400m-patch16-256": 1152,
        "google/siglip2-so400m-patch16-384": 1152,
        "google/siglip2-so400m-patch16-512": 1152,
        "google/siglip2-so400m-patch16-naflex": 1152,
        "google/siglip2-giant-opt-patch16-256": 1280,
        "google/siglip2-giant-opt-patch16-384": 1280,
    }

    def __init__(self, model_name="google/siglip2-base-patch16-224", device="cuda"):
        super().__init__()
        self._init_common(model_name, device)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
class FeatureExtractorFactory:
    """Create an extractor given its model‑checkpoint string."""

    @staticmethod
    def create_extractor(model_name: str, device: str = "cuda") -> BaseFeatureExtractor:
        lower = model_name.lower()
        if "openai/clip" in lower:
            return OpenAICLIPFeatureExtractor(model_name, device)
        if "siglip2" in lower:
            return SigLIP2FeatureExtractor(model_name, device)
        if "siglip" in lower:
            return SigLIPFeatureExtractor(model_name, device)
        if lower.startswith(("vit-b", "vit-l", "vit-h")) or "vit-" in lower:
            return OpenCLIPFeatureExtractor(model_name, device)
        raise ValueError(f"Unsupported extractor type: {model_name}")
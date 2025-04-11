import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List

import logging
logging.getLogger('transformers').setLevel(logging.DEBUG)


class BaseFeatureExtractor(nn.Module, ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.
        
        Args:
            images: Tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        pass

    @abstractmethod
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract features from text.
        
        Args:
            texts: List of strings
            
        Returns:
            Tensor of shape [batch_size, feature_dim]
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of the features."""
        pass

class OpenAICLIPFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using OpenAI CLIP via Hugging Face."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
        self.model = CLIPModel.from_pretrained(model_name)
        
        # Create processor with rescaling disabled
        image_processor = CLIPImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            image_processor=image_processor
        )

        if hasattr(self.model, "text_projection"):
            self._feature_dim = self.model.text_projection.out_features
        else:
            self._feature_dim = self.model.config.projection_dim

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    @property
    def feature_dim(self) -> int:
        return 768 #self._feature_dim


class OpenCLIPFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using OpenCLIP."""

    def __init__(self, model_name: str = "ViT-B-32", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        import open_clip
        self.model, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._feature_dim = self.model.visual.output_dim

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        image_features = self.model.encode_image(images)
        return F.normalize(image_features, dim=-1)

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        return F.normalize(text_features, dim=-1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


# -----------------------------------------------------------------------------
# SigLIPFeatureExtractor for the original SigLIP model.
# -----------------------------------------------------------------------------

class SigLIPFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor based on the original SigLIP model."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "cuda"):
        """
        Args:
            model_name: Name of the SigLIP model variant.
            device: Device to run the model on.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device

        try:
            from transformers import CLIPProcessor, CLIPModel
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            self.model.eval()
            # Set feature dimension based on the available attributes.
            if hasattr(self.model, "text_projection"):
                self._feature_dim = self.model.text_projection.out_features
            else:
                self._feature_dim = self.model.config.projection_dim
        except (ImportError, OSError) as e:
            raise ImportError("Could not load SigLIP model. Please ensure that the transformers library is installed.") from e

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using SigLIP's image encoder."""
        with torch.no_grad():
            images = images.to(self.device)
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract text features using SigLIP's text encoder."""
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

# -----------------------------------------------------------------------------
# SigLIP2FeatureExtractor for the new SigLIP 2 model.
# -----------------------------------------------------------------------------

class SigLIP2FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor based on SigLIP 2 model.

    This implementation uses Hugging Face's AutoProcessor and AutoModel for more flexibility
    in handling different model architectures and versions.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", device: str = "cuda"):
        """
        Args:
            model_name: Name of the SigLIP 2 model variant.
            device: Device to run the model on.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device

        try:
            # Use AutoModel instead of specific Siglip2Model for better compatibility
            from transformers import AutoProcessor, AutoModel, AutoConfig
            
            # Get model configuration first to check compatibility
            self.config = AutoConfig.from_pretrained(model_name)
            print(f"Loaded model config: {self.config.__class__.__name__}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,  # Important for some new model types
                ignore_mismatched_sizes=True  # Try to handle parameter size mismatches
            ).to(device)
            self.model.eval()
            
            # Check for expected methods and adapt if they don't exist
            if not hasattr(self.model, "get_image_features"):
                # Define a wrapper method if the original doesn't exist
                # Original may use a different approach (like direct model calls)
                def get_image_features(self, **kwargs):
                    outputs = self.model.vision_model(**kwargs)
                    return outputs.pooler_output
                
                # Bind this method to the model instance
                from types import MethodType
                self.model.get_image_features = MethodType(get_image_features, self.model)
                
            if not hasattr(self.model, "get_text_features"):
                def get_text_features(self, **kwargs):
                    outputs = self.model.text_model(**kwargs)
                    return outputs.pooler_output
                    
                from types import MethodType
                self.model.get_text_features = MethodType(get_text_features, self.model)
            
            # Set feature dimension based on model configuration
            if hasattr(self.model, "text_projection") and hasattr(self.model.text_projection, "out_features"):
                self._feature_dim = self.model.text_projection.out_features
            elif hasattr(self.config, "projection_dim"):
                self._feature_dim = self.config.projection_dim
            elif hasattr(self.config, "hidden_size"):
                self._feature_dim = self.config.hidden_size
            else:
                # Fallback: try to infer from model architecture
                # This is a guess - may need adjustment
                self._feature_dim = 768  # Common dimension for many models
                print(f"Warning: Could not determine feature dimension, using default: {self._feature_dim}")
                
        except Exception as e:
            raise ImportError(f"Could not load SigLIP 2 model: {str(e)}. "
                             "Please ensure the transformers library is up-to-date and "
                             "the model is available on Hugging Face.") from e

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using SigLIP 2's image encoder."""
        with torch.no_grad():
            images = images.to(self.device)
            try:
                # Try standard approach first
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
            except (AttributeError, TypeError) as e:
                # Fallback if standard approach fails
                print(f"Warning: Standard image feature extraction failed, trying fallback: {str(e)}")
                if hasattr(self.model, "vision_model"):
                    # Process manually if needed
                    if isinstance(images, list):
                        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                    else:
                        inputs = {"pixel_values": images}
                    outputs = self.model.vision_model(**inputs)
                    image_features = outputs.pooler_output
                else:
                    raise ValueError("Cannot extract image features: model structure not compatible")
                
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract text features using SigLIP 2's text encoder."""
        with torch.no_grad():
            try:
                # Try standard approach first
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
            except (AttributeError, TypeError) as e:
                # Fallback if standard approach fails
                print(f"Warning: Standard text feature extraction failed, trying fallback: {str(e)}")
                if hasattr(self.model, "text_model"):
                    # Process manually if needed
                    inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = self.model.text_model(**inputs)
                    text_features = outputs.pooler_output
                else:
                    raise ValueError("Cannot extract text features: model structure not compatible")
                
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

class FeatureExtractorFactory:
    """Factory for creating feature extractors."""

    @staticmethod
    def create_extractor(extractor_type: str, provider: str = "openai", **kwargs) -> BaseFeatureExtractor:
        """
        Create a feature extractor based on the type.

        Args:
            extractor_type: One of ['clip', 'siglip', 'siglip2']
            provider: If extractor_type is 'clip', choose 'openai' or 'openclip'
            **kwargs: Additional arguments passed to the extractor

        Returns:
            BaseFeatureExtractor

        Raises:
            ValueError
        """
        extractor_type = extractor_type.lower()
        provider = provider.lower()

        if extractor_type == 'clip':
            if provider == 'openai':
                return OpenAICLIPFeatureExtractor(**kwargs)
            elif provider == 'openclip':
                return OpenCLIPFeatureExtractor(**kwargs)
            else:
                raise ValueError(f"Unknown CLIP provider: {provider}")
        elif extractor_type == 'siglip':
            return SigLIPFeatureExtractor(**kwargs)
        elif extractor_type == 'siglip2':
            return SigLIP2FeatureExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported feature extractor type: {extractor_type}")

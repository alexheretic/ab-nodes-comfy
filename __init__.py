import hashlib
import logging
import os

import folder_paths
import nodes
import torch

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def touch_cache_file(path):
    """Update a cache file's mtime to mark it as recently used."""
    try:
        os.utime(path, None)
    except Exception:
        logger.warning("Failed to touch %s", path, exc_info=True)


def prune_cache(cache_dir, max_items):
    """Keep only the max_items most recently modified *.pt files in cache_dir."""
    try:
        entries = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ]
    except FileNotFoundError:
        return

    if len(entries) <= max_items:
        return

    entries.sort(key=lambda p: os.path.getmtime(p))
    for path in entries[: len(entries) - max_items]:
        try:
            os.remove(path)
            logger.info("Pruned: %s", os.path.basename(path))
        except Exception:
            logger.warning("Failed to prune %s", path, exc_info=True)


class CLIPTextEncodeDiskCache(nodes.CLIPTextEncode):
    """CLIPTextEncode with disk cache of the most recently used."""

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        return base_inputs

    RETURN_TYPES = nodes.CLIPTextEncode.RETURN_TYPES
    RETURN_NAMES = getattr(nodes.CLIPTextEncode, "RETURN_NAMES", ("CONDITIONING",))
    FUNCTION = "encode"
    CATEGORY = "conditioning/custom"

    MAX_CACHE_ITEMS = 30
    CACHE_DIR = os.path.join(
        folder_paths.get_user_directory(), "ab-nodes", "clip_text_encode"
    )

    def _clip_hash(self, clip):
        try:
            model = getattr(clip, "cond_stage_model", None) or clip
            state_dict = model.state_dict()

            hasher = hashlib.sha256()
            # Hashing every tensor would be slow for large text encoders,
            # so sample a fixed, deterministically-ordered subset. Sorting
            # keys guarantees the same tensors are picked every time.
            keys = sorted(state_dict.keys())
            sample_keys = keys[:3] + keys[-3:] if len(keys) > 6 else keys

            for k in sample_keys:
                tensor = state_dict[k].detach().cpu().contiguous()
                hasher.update(k.encode("utf-8"))
                hasher.update(tensor.numpy().tobytes())

            return hasher.hexdigest()[:16]
        except Exception:
            logger.warning("clip_hash failed falling back to id()", exc_info=True)
            return f"fallback-{id(clip)}"

    def _cache_key(self, clip, text):
        clip_hash = self._clip_hash(clip)
        raw_key = f"{clip_hash}::{text}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.CACHE_DIR, f"{key}.pt")

    def encode(self, clip, text):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        key = self._cache_key(clip, text)
        cache_path = self._cache_path(key)

        if os.path.exists(cache_path):
            try:
                conditioning = torch.load(cache_path, weights_only=False)
                touch_cache_file(cache_path)
                logger.info("CLIPTextEncodeDiskCache: Cache hit: %s", key)
                return (conditioning,)
            except Exception:
                logger.warning(
                    "CLIPTextEncodeDiskCache: Cache load failed for %s, regenerating.", key, exc_info=True
                )

        logger.info("CLIPTextEncodeDiskCache: Cache miss: %s", key)
        result = super().encode(clip, text)
        conditioning = result[0]

        try:
            torch.save(conditioning, cache_path)
            prune_cache(self.CACHE_DIR, self.MAX_CACHE_ITEMS)
        except Exception:
            logger.warning("CLIPTextEncodeDiskCache: Failed to write cache for %s", key, exc_info=True)

        return (conditioning,)


NODE_CLASS_MAPPINGS["CLIPTextEncodeDiskCache"] = CLIPTextEncodeDiskCache
NODE_DISPLAY_NAME_MAPPINGS["CLIPTextEncodeDiskCache"] = "CLIP Text Encode (Disk Cache)"


class CLIPVisionEncodeDiskCache(nodes.CLIPVisionEncode):
    """CLIPVisionEncode with disk cache of the most recently used."""

    CACHE_DIR = os.path.join(
        folder_paths.get_user_directory(), "ab-nodes", "clip_vision_encode"
    )
    MAX_CACHE_ITEMS = 30

    RETURN_TYPES = nodes.CLIPVisionEncode.RETURN_TYPES
    FUNCTION = "encode"
    CATEGORY = "conditioning/custom"

    def _clip_vision_hash(self, clip_vision):
        try:
            model = getattr(clip_vision, "model", None) or clip_vision
            state_dict = model.state_dict()

            hasher = hashlib.sha256()
            keys = sorted(state_dict.keys())
            sample_keys = keys[:3] + keys[-3:] if len(keys) > 6 else keys

            for k in sample_keys:
                tensor = state_dict[k].detach().cpu().contiguous()
                hasher.update(k.encode("utf-8"))
                hasher.update(tensor.numpy().tobytes())

            return hasher.hexdigest()[:16]
        except Exception:
            logger.warning(
                "clip_vision fingerprint failed, falling back to id()", exc_info=True
            )
            return f"fallback-{id(clip_vision)}"

    def _image_hash(self, image):
        tensor = image.detach().cpu().contiguous()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    def _cache_key(self, clip_vision, image, crop):
        clip_vision_hash = self._clip_vision_hash(clip_vision)
        image_hash = self._image_hash(image)
        raw_key = f"{clip_vision_hash}::{image_hash}::{crop}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.CACHE_DIR, f"{key}.pt")

    def encode(self, clip_vision, image, crop="center"):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        key = self._cache_key(clip_vision, image, crop)
        cache_path = self._cache_path(key)

        if os.path.exists(cache_path):
            try:
                output = torch.load(cache_path, weights_only=False)
                touch_cache_file(cache_path)
                logger.info("CLIPVisionEncodeDiskCache: Cache hit: %s", key)
                return (output,)
            except Exception:
                logger.warning(
                    "CLIPVisionEncodeDiskCache: Cache load failed for %s, regenerating.", key, exc_info=True
                )

        logger.info("CLIPVisionEncodeDiskCache: Cache miss: %s", key)
        result = super().encode(clip_vision, image, crop=crop)
        output = result[0]

        try:
            torch.save(output, cache_path)
            prune_cache(self.CACHE_DIR, self.MAX_CACHE_ITEMS)
        except Exception:
            logger.warning("CLIPVisionEncodeDiskCache: Failed to write cache for %s", key, exc_info=True)

        return (output,)


NODE_CLASS_MAPPINGS["CLIPVisionEncodeDiskCache"] = CLIPVisionEncodeDiskCache
NODE_DISPLAY_NAME_MAPPINGS["CLIPVisionEncodeDiskCache"] = (
    "CLIP Vision Encode (Disk Cache)"
)

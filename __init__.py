import hashlib
import logging
import os

import folder_paths
import nodes
import torch

logger = logging.getLogger(__name__)

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

    def _touch(self, path):
        """Update both atime and mtime to now, marking this entry as recently used."""
        try:
            os.utime(path, None)
        except Exception:
            logger.warning("Failed to touch %s", path, exc_info=True)

    def _prune_cache(self):
        """Keep only the MAX_CACHE_ITEMS most recently modified cache files."""
        try:
            entries = [
                os.path.join(self.CACHE_DIR, f)
                for f in os.listdir(self.CACHE_DIR)
                if f.endswith(".pt")
            ]
        except FileNotFoundError:
            return

        if len(entries) <= self.MAX_CACHE_ITEMS:
            return

        entries.sort(key=lambda p: os.path.getmtime(p))
        num_to_remove = len(entries) - self.MAX_CACHE_ITEMS

        for path in entries[:num_to_remove]:
            try:
                os.remove(path)
                logger.debug("Pruned: %s", os.path.basename(path))
            except Exception:
                logger.warning("Failed to prune %s", path, exc_info=True)

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
                self._touch(cache_path)
                logger.debug("Cache hit: %s", key)
                return (conditioning,)
            except Exception:
                logger.warning("Cache load failed for %s, regenerating.", key, exc_info=True)


        logger.debug("Cache miss: %s", key)
        result = super().encode(clip, text)
        conditioning = result[0]

        try:
            torch.save(conditioning, cache_path)
            self._prune_cache()
        except Exception:
            logger.warning("Failed to write cache for %s", key, exc_info=True)

        return (conditioning,)


NODE_CLASS_MAPPINGS = {"CLIPTextEncodeDiskCache": CLIPTextEncodeDiskCache}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeDiskCache": "CLIP Text Encode (Disk Cache)"
}

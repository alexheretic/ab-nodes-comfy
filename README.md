# ab-nodes-comfy
Custom nodes for ComfyUI.

* **CLIPTextEncodeDiskCache**: CLIPTextEncode with a disk cache of the 30 most recently used encodes.
  Stored in `./user/ab-nodes/clip_text_encode`. This can help prevent the text encoder model being loaded
  into VRAM when encoding the same prompts multiple times.
* **CLIPVisionEncodeDiskCache**: CLIPVisionEncode with a disk cache of the 30 most recently used encodes.
  Stored in `./user/ab-nodes/clip_vision_encode`. This can help prevent the clip vision model being loaded
  into VRAM when encoding the same images multiple times.

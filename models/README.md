# Model weights

Weights are local and ignored by Git. The default runtime automatically downloads
`yolo26n.pt` from Ultralytics when needed. Qwen uses the Hugging Face cache.
Use `scripts/runtime/download_models.py --qwen` to prefetch both.

Private traffic checkpoints may be stored here and selected in a copied TaskConfig.
They must match the detector class IDs configured for the task. No sports-specific
checkpoint is required by the public traffic profile.

# sitecustomize.py
try:
    from wandb.integration.gym import RecordVideo
    # ensure the flag exists so RecordVideo.close() won’t crash
    RecordVideo.enabled = True
except Exception:
    pass


# FishAudio S2 Pro Modal App

This folder isolates FishAudio S2 Pro experiments from the main `modal/app.py` pipeline.

Run the fixed SP_SP010 Hindi demo:

```bash
cd /home/pranav/TINYYAYAy/llama-mimi
source .venv/bin/activate
modal run --detach modal/fish_audio_s2_pro/app.py::sp_sp010_hindi_demo
```

The app writes artifacts to:

```text
/vol/outputs/fish_audio_s2_pro/sp_sp010_hindi_demo
```

Expected files after a successful run:

- `devanagari_hindi.wav`
- `latin_hindi.wav`
- `metadata.json`
- `server.log`

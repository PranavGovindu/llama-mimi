# Modal

This directory holds Modal-specific launchers and environment probes for the
TinyAya TTS Lab.

Use these files as the active entrypoints:
- `train_fa3.py`
  - slim Hopper training path with the kernels-based FA3 backend
- `app.py`
  - broader legacy launcher with dataset prep, export, and mixed workflows
- `fa3_probe.py`
  - attention backend smoke/benchmark utilities
- `ngc_stack_probe.py`
  - native NGC environment sanity probe

Guidelines:
- keep model/runtime logic in `torchtitan/`, not here
- keep canonical configs in `recipes/`
- treat Modal files as thin launch wrappers around repo-native commands

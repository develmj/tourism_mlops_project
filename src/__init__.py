"""
Package initializer for the tourism_mlops_project.src module.

This file allows Python (locally, in Colab, and in GitHub Actions) to treat
the 'src' directory as an importable package if needed.

No runtime logic is placed here intentionally — the training pipeline should
always be executed via:

    python src/train_pipeline.py

This file simply ensures clean packaging and avoids import errors.
"""


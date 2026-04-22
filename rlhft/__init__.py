from pathlib import Path

# Expose the src-layout package when running from the repository root
# without requiring an editable install first.
__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "rlhft")]

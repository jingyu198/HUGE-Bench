from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("3dgs_renderer.py")), run_name="__main__")

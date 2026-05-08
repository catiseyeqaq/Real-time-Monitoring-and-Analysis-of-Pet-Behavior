from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent


def main():
    model = YOLO(ROOT_DIR / "yolo26n.pt")
    model.train(
        data=str(ROOT_DIR / "猫咪行为数据集" / "cat_behavior_merged" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        project=str(ROOT_DIR / "runs" / "train"),
        name="cat_behavior_yolo26n",
    )


if __name__ == "__main__":
    freeze_support()
    main()

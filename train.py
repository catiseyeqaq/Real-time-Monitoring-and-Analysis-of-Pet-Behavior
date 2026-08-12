"""Train a local behavior model from an annotated pet dataset."""

from __future__ import annotations

import os
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO as BehaviorModel


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = ROOT_DIR / "models" / "pet_behavior_base.pt"
DEFAULT_DATA_CONFIG = ROOT_DIR / "猫咪行为数据集" / "cat_behavior_merged" / "data.yaml"


def main() -> None:
    model_path = Path(os.environ.get("PET_BEHAVIOR_BASE_MODEL", str(DEFAULT_BASE_MODEL)))
    data_config = Path(os.environ.get("PET_BEHAVIOR_DATA_CONFIG", str(DEFAULT_DATA_CONFIG)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"基础模型不存在: {model_path}\n"
            "请通过 PET_BEHAVIOR_BASE_MODEL 指定可用模型，或直接使用已有权重运行应用。"
        )
    if not data_config.exists():
        raise FileNotFoundError(
            f"数据配置不存在: {data_config}\n"
            "请通过 PET_BEHAVIOR_DATA_CONFIG 指定标注数据配置文件。"
        )

    model = BehaviorModel(str(model_path))
    model.train(
        data=str(data_config),
        epochs=int(os.environ.get("PET_BEHAVIOR_EPOCHS", "100")),
        imgsz=int(os.environ.get("PET_BEHAVIOR_IMAGE_SIZE", "640")),
        batch=int(os.environ.get("PET_BEHAVIOR_BATCH", "16")),
        workers=int(os.environ.get("PET_BEHAVIOR_WORKERS", "4")),
        project=str(ROOT_DIR / "runs" / "train"),
        name="pet_behavior",
    )


if __name__ == "__main__":
    freeze_support()
    main()

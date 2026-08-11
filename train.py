"""Train a local pet behavior model from an annotated dataset.

This optional script is a training helper for project-owned behavior classes.
The trained weight is intentionally ignored by Git and must be managed by the
deployment owner.
"""

from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO as BehaviorModel


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = ROOT_DIR / "models" / "pet_behavior_base.pt"
DEFAULT_DATA_CONFIG = ROOT_DIR / "猫咪行为数据集" / "cat_behavior_merged" / "data.yaml"


def main() -> None:
    model_path = Path(__import__("os").environ.get("PET_BEHAVIOR_BASE_MODEL", DEFAULT_BASE_MODEL))
    data_config = Path(__import__("os").environ.get("PET_BEHAVIOR_DATA_CONFIG", DEFAULT_DATA_CONFIG))
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
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        project=str(ROOT_DIR / "runs" / "train"),
        name="pet_behavior",
    )


if __name__ == "__main__":
    freeze_support()
    main()

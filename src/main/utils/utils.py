import os
from enum import Enum, auto
from pathlib import Path
from attr import dataclass
from sklearn.ensemble import ExtraTreesRegressor
import torch

# TabPFN needs a token for the one time model download + license check.
# This one is read only (inference only), so fine to keep in the repo, dont need to hide behind env vars.
# An env-provided TABPFN_TOKEN still takes precedence so token rotation can be
# handled without a repo edit.
TABPFN_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYzk1YTBjNGMtNDZjYS00MjhiLTgyNDQtNWRlMWNhNGJkZTdkIiwiZXhwIjoxODA3NjM3MzMyfQ.4V6VuHGT9OEHg1lzLr8lEM411T6IHMCuEg1j1yWfo10"
os.environ.setdefault("TABPFN_TOKEN", TABPFN_TOKEN)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DataType(Enum):
    TRAIN = auto()
    VAL = auto()


class ModelType(Enum):
    EXTRA_TREES = auto()
    TABPFN = auto()

    @classmethod
    def from_string(cls, name: str) -> "ModelType":
        # Used to map the --model CLI arg to the enum value.
        mapping = {"tabpfn": cls.TABPFN, "extra_trees": cls.EXTRA_TREES}
        key = name.lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown model '{name}'. Valid options: {list(mapping.keys())}"
            )
        return mapping[key]

class VisionBackbone(str, Enum):
    DINO = "dino"
    RESNET = "resnet"
    CONVNEXT = "convnext"

    @classmethod
    def from_string(cls, name: str) -> "VisionBackbone":
        try:
            return cls(name.lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown vision backbone '{name}'. Valid options: {[m.value for m in cls]}"
            ) from exc

    def __str__(self) -> str:
        return self.value


def _normalize_vision_backbone(
    backbone: VisionBackbone | str,
) -> VisionBackbone:
    if isinstance(backbone, VisionBackbone):
        return backbone
    return VisionBackbone.from_string(backbone)


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42

    # CV
    n_splits: int = 5
    n_jobs: int = -1

    # models
    pca_n_components: int = 128
    model_type: ModelType = ModelType.EXTRA_TREES

    # resources
    lower_resources: bool = True
    max_cv_groups: int = 160

    device: str = _auto_device()

    # targets
    TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
    MANUAL_TARGETS = ["Dry_Total_g", "GDM_g"]

    def get_model(self):
        if self.model_type == ModelType.TABPFN:
            # Lazy import so runs with --model extra_trees don't pay the tabpfn import cost.
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor(random_state=self.random_state)

        return ExtraTreesRegressor(
            n_estimators=275,
            min_samples_leaf=3,
            max_depth=None,
            n_jobs=-1,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class VisionModelConfig:
    vision_backbone: VisionBackbone | str = VisionBackbone.DINO
    image_size: int = 350
    batch_size: int = 64
    num_workers: int = 0

    device: str = _auto_device()

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "vision_backbone",
            _normalize_vision_backbone(self.vision_backbone),
        )

    @property
    def model_name(self) -> str:
        if self.vision_backbone == VisionBackbone.DINO:
            return "dinov2_vits14"

        raise ValueError(
            f"No model_name is defined for vision backbone '{self.vision_backbone}'."
        )

    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DatasetPaths:
    def __attrs_post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

    root: Path = Path(__file__).parent.parent.parent / "data"
    model_dir: Path = Path(__file__).parent.parent / "model_data"

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"

    def _vision_feats_path(
        self,
        split: str,
        backbone: VisionBackbone | str,
        image_size: int | None,
    ) -> Path:
        name = f"feature_{split}_{_normalize_vision_backbone(backbone)}"
        if image_size is not None:
            name += f"_s{image_size}"
        return self.model_dir / f"{name}.npy"

    def vision_feats_train_path(
        self,
        backbone: VisionBackbone | str = VisionBackbone.DINO,
        image_size: int | None = None,
    ) -> Path:
        return self._vision_feats_path("train", backbone, image_size)

    def vision_feats_test_path(
        self,
        backbone: VisionBackbone | str = VisionBackbone.DINO,
        image_size: int | None = None,
    ) -> Path:
        return self._vision_feats_path("test", backbone, image_size)

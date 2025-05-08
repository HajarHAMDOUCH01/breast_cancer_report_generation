import tempfile
from pathlib import Path
from torchvision.datasets.utils import download_url
from .image_model import BreastCancerImageModel
from .types import ImageEncoderType

JOINT_FEATURE_SIZE = 128

# Hugging Face repo identifiers and file names
BIOMED_VLP_BIOVIL_T = "microsoft/BiomedVLP-BioViL-T"
HF_URL = "https://huggingface.co"
BIOVIL_T_COMMIT_TAG = "v1.0"
BIOVIL_T_IMAGE_WEIGHTS_NAME = "biovil_t_image_model_proj_size_128.pt"
BIOVIL_T_IMAGE_WEIGHTS_URL = f"{HF_URL}/{BIOMED_VLP_BIOVIL_T}/resolve/{BIOVIL_T_COMMIT_TAG}/{BIOVIL_T_IMAGE_WEIGHTS_NAME}"
BIOVIL_T_IMAGE_WEIGHTS_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"

def download_biovil_t_weights(cache_dir: Path | None = None) -> Path:
    """
    Download BioViL-T pretrained weights from Hugging Face and return local path.
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_url(
        BIOVIL_T_IMAGE_WEIGHTS_URL,
        root=str(cache_dir),
        filename=BIOVIL_T_IMAGE_WEIGHTS_NAME,
        md5=BIOVIL_T_IMAGE_WEIGHTS_MD5,
    )
    return cache_dir / BIOVIL_T_IMAGE_WEIGHTS_NAME


def get_biovil_t_image_encoder(pretrained: bool = True) -> BreastCancerImageModel:
    """
    Instantiate BioViL-T image encoder. Downloads weights if pretrained=True.
    """
    ckpt_path = download_biovil_t_weights("./pretrained_biovil_t_weights") if pretrained else None
    model = BreastCancerImageModel(
        img_encoder_type=ImageEncoderType.BIOVIL_T_MULTI,
        joint_feature_size=JOINT_FEATURE_SIZE,
        freeze_backbone=False,
        pretrained_model_path=ckpt_path,
    )
    return model

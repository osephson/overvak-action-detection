import dataclasses
import os
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from pytube import YouTube
from torchvision.models.video import R3D_18_Weights, r3d_18
import yt_dlp
from yt_dlp.utils import DownloadError


TARGET_ACTIONS = [
    "climbing",
    "crying",
    "falling",
    "hitting",
    "jumping",
    "laughing",
    "pushing",
    "running",
]


@dataclasses.dataclass
class ActionPrediction:
    action: str
    score: float


@dataclasses.dataclass
class VideoResult:
    video_id: str
    kinetics_topk: List[Tuple[str, float]]
    mapped_actions: List[ActionPrediction]


def _load_model() -> Tuple[torch.nn.Module, R3D_18_Weights]:
    """Load r3d_18 Kinetics-400 model and weights once per process."""
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.eval()
    return model, weights


_MODEL = None
_WEIGHTS = None


def get_model_and_weights() -> Tuple[torch.nn.Module, R3D_18_Weights]:
    global _MODEL, _WEIGHTS
    if _MODEL is None or _WEIGHTS is None:
        _MODEL, _WEIGHTS = _load_model()
    return _MODEL, _WEIGHTS


def download_youtube_video(url: str) -> Tuple[str, str]:
    """Download a YouTube video and return (video_path, video_id).

    Uses yt-dlp for robustness and falls back to pytube if needed.
    """
    tmp_dir = tempfile.mkdtemp(prefix="overvak_")

    # Try yt-dlp first – it tends to be more robust to YouTube changes.
    try:
        ydl_opts = {
            "format": "mp4[height<=360]/mp4",
            "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id")
            filename = ydl.prepare_filename(info)
        return filename, video_id
    except DownloadError as e:
        msg = str(e)
        # Surface a clear message for common bot / sign-in checks.
        if "Sign in to confirm you’re not a bot" in msg or "not a bot" in msg:
            raise RuntimeError(
                "YouTube is asking for sign‑in / bot confirmation for this video. "
                "Please either use a different public video URL or download the "
                "video manually in your browser and pass its local file path "
                "into the app instead of the URL."
            )
        # Otherwise fall back to pytube below.
    except Exception:
        # Fallback to pytube if yt-dlp fails for any reason.
        yt = YouTube(url)
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .first()
        )
        if stream is None:
            raise RuntimeError("No suitable mp4 stream found for this YouTube URL.")

        out_path = stream.download(output_path=tmp_dir, filename="video.mp4")
        return out_path, yt.video_id


def _sample_frames(
    video_path: str, num_frames: int = 16
) -> torch.Tensor:
    """Uniformly sample `num_frames` frames and return a (T, C, H, W) tensor."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video appears to have no frames.")

    indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
    frames: List[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("Failed to read any frames from video.")

    # Convert to torch tensor in (T, C, H, W)
    arr = np.stack(frames)  # (T, H, W, C)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    return video


def infer_actions_from_video(
    video_path: str,
    topk_kinetics: int = 5,
) -> VideoResult:
    """Run the action model on a single video file and map to target actions."""
    model, weights = get_model_and_weights()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    video = _sample_frames(video_path, num_frames=16)
    # Apply Kinetics preprocessing. Returns (C, T, H, W).
    video = preprocess(video)
    # Add batch dimension -> (1, C, T, H, W).
    video = video.unsqueeze(0)

    with torch.no_grad():
        logits = model(video)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1][:topk_kinetics]
    kinetics_top = [(categories[i], float(probs[i])) for i in top_indices]

    mapped = map_kinetics_to_target_actions(kinetics_top)

    return VideoResult(
        video_id=os.path.splitext(os.path.basename(video_path))[0],
        kinetics_topk=kinetics_top,
        mapped_actions=mapped,
    )


def map_kinetics_to_target_actions(
    kinetics_predictions: List[Tuple[str, float]],
) -> List[ActionPrediction]:
    """Map generic Kinetics action labels to the child-centric target actions.

    The mapping is intentionally simple and heuristic; for a production system
    we'd train or fine-tune a model directly on the target label space.
    """
    # Simple substring based mapping rules from Kinetics labels.
    rules: Dict[str, List[str]] = {
        "climbing": ["climbing", "rock climbing", "ladder"],
        "crying": ["crying", "sobbing"],
        "falling": ["falling", "falling down"],
        "hitting": ["punching", "slapping", "hitting"],
        "jumping": ["jumping", "leapfrog"],
        "laughing": ["laughing"],
        "pushing": ["pushing"],
        "running": ["running", "jogging", "sprinting"],
    }

    scores: Dict[str, float] = {a: 0.0 for a in TARGET_ACTIONS}

    for label, prob in kinetics_predictions:
        label_l = label.lower()
        for action, keywords in rules.items():
            if any(kw in label_l for kw in keywords):
                scores[action] = max(scores[action], prob)

    results = [
        ActionPrediction(action=a, score=s)
        for a, s in scores.items()
        if s > 0.0
    ]
    # Sort by score descending.
    results.sort(key=lambda x: x.score, reverse=True)
    return results


"""
Documentation-only companion to video_rag.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with video_rag.py
automatically.
"""


# --8<-- [start:extract_frames]
def extract_frames(video_path: str, interval_sec: int = 5) -> list:
    """Sample frames at a fixed interval via ffmpeg -- the only
    "extraction" step needed on the visual side. No scene detection, no
    frame-importance scoring; every Nth second gets a frame regardless of
    whether anything changed."""
    import subprocess
    import tempfile
    from pathlib import Path

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vf", f"fps=1/{interval_sec}", "-y", f"{tmp}/frame_%04d.jpg"],
            check=True, capture_output=True,
        )
        frames = []
        for i, path in enumerate(sorted(Path(tmp).glob("frame_*.jpg"))):
            frames.append((i * interval_sec, Image.open(path).copy()))
    return frames
# --8<-- [end:extract_frames]


# --8<-- [start:transcribe_video]
def transcribe_video(video_path: str, model_size: str = "small") -> list:
    """Speech-to-text with timestamped segments. Whisper shells out to
    the system `ffmpeg` binary internally to decode audio, so ffmpeg is a
    real prerequisite here even though this function never calls it
    directly."""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]
# --8<-- [end:transcribe_video]


# --8<-- [start:transcript_search]
def transcript_search(query: str, segments: list, k: int = 3) -> list:
    """Embed each transcript segment once, then rank by cosine
    similarity to the query -- the same pattern used for text chunks
    elsewhere on this site, just with a timestamp attached to each
    unit instead of a page or document id."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [s["text"] for s in segments]
    segment_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, segment_embeddings)[0]
    top_k = sims.topk(min(k, len(segments)))
    return [segments[i] for i in top_k.indices.tolist()]
# --8<-- [end:transcript_search]


# --8<-- [start:nearest_frame]
def nearest_frame(timestamp: float, frames: list):
    """Frames only exist at the fixed sampling interval, so the frame
    actually shown alongside a retrieved transcript segment is whichever
    sampled frame is CLOSEST to that segment's start time -- not
    necessarily the exact right visual moment."""
    return min(frames, key=lambda pair: abs(pair[0] - timestamp))[1]
# --8<-- [end:nearest_frame]

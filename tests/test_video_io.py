import gc
import json

import cv2
import numpy as np
import pytest

from src.types import FrameState
from src.video_io import ControlLogWriter, VideoReader, VideoWriter


@pytest.fixture
def clip(tmp_path):
    """A 10-frame 64x48 video at 10 fps, frame i filled with intensity i*10."""
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    return str(p)


def test_reader_reads_every_frame(clip):
    r = VideoReader(clip)
    frames = list(r)
    assert len(frames) == 10
    assert r.width == 64 and r.height == 48
    assert r.fps == pytest.approx(10.0, abs=0.1)


def test_reader_stride_skips_frames(clip):
    frames = list(VideoReader(clip, stride=3))
    assert [i for i, _, _ in frames] == [0, 3, 6, 9]


def test_reader_timestamps_use_source_fps(clip):
    frames = list(VideoReader(clip, stride=2))
    assert [round(t, 3) for _, t, _ in frames] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_reader_max_frames(clip):
    assert len(list(VideoReader(clip, max_frames=4))) == 4


def test_writer_roundtrip(clip, tmp_path):
    out = tmp_path / "out.mp4"
    with VideoWriter(str(out), 10.0, (64, 48)) as w:
        for _, _, f in VideoReader(clip):
            w.write(f)
    assert out.exists() and out.stat().st_size > 0
    assert len(list(VideoReader(str(out)))) == 10


def test_control_log_writes_one_json_per_frame(tmp_path):
    img = np.zeros((48, 64, 3), np.uint8)
    p = tmp_path / "log.jsonl"
    with ControlLogWriter(str(p)) as log:
        for i in range(3):
            s = FrameState(idx=i, t=i * 0.1, raw=img, img=img)
            s.control.steer_deg = float(i)
            s.decision.log.append(f"event {i}")
            log.write(s)

    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[1]["steer_deg"] == 1.0
    assert rows[2]["events"] == ["event 2"]
    assert rows[0]["fsm_state"] == "LANE_KEEP"


def test_reader_releases_capture_when_iterator_abandoned(clip):
    r = VideoReader(clip)
    it = iter(r)
    next(it)
    next(it)
    assert r._cap.isOpened()
    del it
    gc.collect()
    assert not r._cap.isOpened(), "capture must be released when the iterator is abandoned"


def test_reader_releases_capture_on_consumer_exception(clip):
    r = VideoReader(clip)
    with pytest.raises(RuntimeError):
        for _ in r:
            raise RuntimeError("consumer blew up")
    assert not r._cap.isOpened()

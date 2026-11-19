"""Check actual MP4 encoding and truthful bilingual experiment reports."""

import json
import math

import imageio.v2 as imageio
import numpy as np
import pytest

from agile.controllers.experiment_artifacts import (
    ExperimentVideo,
    create_experiment_directory,
    validate_video,
    write_experiment_data,
    write_experiment_reports,
)


def test_experiments_stay_in_outputs_and_never_overwrite(tmp_path):
    output = create_experiment_directory(tmp_path, "trial")
    assert output == tmp_path / "outputs" / "trial"
    with pytest.raises(FileExistsError):
        create_experiment_directory(tmp_path, "outputs/trial")
    with pytest.raises(ValueError):
        create_experiment_directory(tmp_path, "../outside")
    a = create_experiment_directory(tmp_path)
    b = create_experiment_directory(tmp_path)
    assert a != b


def test_video_is_decodable_and_preserves_phase_timing(tmp_path):
    video = ExperimentVideo(tmp_path / "video.mp4", 25)
    for i in range(25):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 10
        video.append(frame, "prepare" if i < 5 else "rollout")
    summary = video.close()
    assert summary["status"] == "saved"
    assert summary["duration_s"] == 1
    assert summary["validation"]["decoded_frames"] == 25
    assert summary["validation"]["entirely_black_frame_indices"] == [0]
    assert summary["phases"][1] == {"phase": "rollout", "start_s": 0.2, "frames": 20}
    reader = imageio.get_reader(tmp_path / "video.mp4")
    try:
        assert reader.count_frames() == 25
        assert reader.get_meta_data()["fps"] == 25
        assert reader.get_data(24)[:, :, 0].mean() > reader.get_data(0)[:, :, 0].mean() + 200
    finally:
        reader.close()
    with pytest.raises(RuntimeError, match="expected 26"):
        validate_video(tmp_path / "video.mp4", 26)


def test_empty_video_is_not_claimed_as_saved(tmp_path):
    video = ExperimentVideo(tmp_path / "video.mp4", 25)
    with pytest.raises(ValueError):
        video.append(None, "rollout")
    assert video.close()["status"] == "missing"


def test_data_only_output_does_not_generate_documents_or_video(tmp_path):
    summary = {"status": "complete", "video": {"status": "disabled"}}
    config = {"command": "data-only run"}
    write_experiment_data(tmp_path, summary, config)
    assert {path.name for path in tmp_path.iterdir()} == {"metrics.json", "run_config.json"}
    assert json.loads((tmp_path / "metrics.json").read_text()) == summary
    assert json.loads((tmp_path / "run_config.json").read_text()) == config


def test_requested_reports_distinguish_disabled_video_from_failed_recording(tmp_path):
    write_experiment_reports(tmp_path, {"status": "complete", "video": {"status": "disabled"}}, {})
    en = (tmp_path / "report_en.md").read_text()
    zh = (tmp_path / "report_zh.md").read_text()
    assert "Video recording was not requested" in en and "Video unavailable" not in en
    assert "本次未请求录制视频" in zh and "录像不可用" not in zh


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_reports_share_measured_values_and_disclose_missing_video(tmp_path, status):
    summary = {
        "status": status,
        "mode": "hold",
        "position_error_m_rms": 0.031,
        "orientation_error_rad_rms": math.pi / 6,
        "failure_reason": "example failure" if status == "failed" else None,
        "video": {"status": "missing"},
    }
    write_experiment_reports(tmp_path, summary, {"command": "example command"})
    en = (tmp_path / "report_en.md").read_text()
    zh = (tmp_path / "report_zh.md").read_text()
    for report in (en, zh):
        assert "3.100" in report
        assert "30.000" in report
        assert "example command" in report
    assert "Video unavailable" in en
    assert "录像不可用" in zh
    if status == "failed":
        assert "experiment failed or ended early" in en
        assert "实验失败或提前结束" in zh
        assert "completed without environment termination" not in en
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == status

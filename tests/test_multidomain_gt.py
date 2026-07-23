import json
from pathlib import Path

import cv2
import numpy as np

from football_tracking.data.multidomain_gt import convert_multidomain_gt


def test_convert_bdd100k_scalabel(tmp_path: Path) -> None:
    source = tmp_path / "bdd.json"
    source.write_text(
        json.dumps(
            [
                {
                    "videoName": "traffic-1",
                    "frameIndex": 0,
                    "labels": [
                        {
                            "id": "car-a",
                            "category": "car",
                            "box2d": {"x1": 1, "y1": 2, "x2": 11, "y2": 22},
                        }
                    ],
                },
                {
                    "videoName": "traffic-1",
                    "frameIndex": 1,
                    "labels": [
                        {
                            "id": "car-a",
                            "category": "car",
                            "box2d": {"x1": 2, "y1": 3, "x2": 12, "y2": 23},
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = convert_multidomain_gt(
        source_format="bdd100k_scalabel",
        annotation_path=source,
        output_dir=tmp_path / "out",
    )

    assert result["sequence_count"] == 1
    assert result["sequences"][0]["track_count"] == 1
    lines = Path(result["sequences"][0]["gt_path"]).read_text().splitlines()
    assert lines[0].startswith("1,1,1.000000,2.000000,10.000000,20.000000")
    assert lines[1].startswith("2,1,2.000000,3.000000,10.000000,20.000000")


def test_convert_tao_coco_video(tmp_path: Path) -> None:
    source = tmp_path / "tao.json"
    source.write_text(
        json.dumps(
            {
                "videos": [{"id": 7, "name": "bird/video"}],
                "images": [{"id": 9, "video_id": 7, "frame_index": 0}],
                "categories": [{"id": 3, "name": "kingfisher"}],
                "annotations": [
                    {"id": 4, "image_id": 9, "track_id": 22, "category_id": 3, "bbox": [5, 6, 7, 8]}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = convert_multidomain_gt(
        source_format="tao_coco_video",
        annotation_path=source,
        output_dir=tmp_path / "out",
    )

    assert result["categories"] == {"3": "kingfisher"}
    assert result["sequences"][0]["normalized_sequence"] == "bird_video"


def test_convert_animaltrack_mot(tmp_path: Path) -> None:
    source = tmp_path / "tiger.txt"
    source.write_text("1,1,10,20,30,40,1,2,0.9\n", encoding="utf-8")
    category_map = tmp_path / "classes.json"
    category_map.write_text(json.dumps({"2": "tiger"}), encoding="utf-8")

    result = convert_multidomain_gt(
        source_format="animaltrack_mot",
        annotation_path=source,
        output_dir=tmp_path / "out",
        category_map_path=category_map,
    )

    assert result["annotation_count"] == 1
    assert result["categories"] == {"2": "tiger"}
    assert Path(result["sequences"][0]["seqinfo_path"]).is_file()


def test_animaltrack_gt_suffix_is_removed(tmp_path: Path) -> None:
    source = tmp_path / "zebra_1_gt.txt"
    source.write_text("1,1,10,20,30,40,1,10,-1\n", encoding="utf-8")

    result = convert_multidomain_gt(
        source_format="animaltrack_mot",
        annotation_path=source,
        output_dir=tmp_path / "out",
    )

    assert result["sequences"][0]["normalized_sequence"] == "zebra_1"
    assert (tmp_path / "out" / "zebra_1" / "gt" / "gt.txt").is_file()


def test_animaltrack_video_metadata_is_written_to_seqinfo(tmp_path: Path) -> None:
    source = tmp_path / "gt" / "zebra_1_gt.txt"
    source.parent.mkdir()
    source.write_text("1,1,10,20,30,40,1,10,-1\n", encoding="utf-8")
    video = _write_test_video(tmp_path / "videos" / "zebra_1.mp4")

    result = convert_multidomain_gt(
        source_format="animaltrack_mot",
        annotation_path=source.parent,
        media_root=video.parent,
        output_dir=tmp_path / "out",
    )

    sequence = result["sequences"][0]
    assert sequence["frame_count"] == 3
    seqinfo = Path(sequence["seqinfo_path"]).read_text(encoding="utf-8")
    assert "seqLength=3" in seqinfo
    assert "imWidth=64" in seqinfo
    assert "imHeight=48" in seqinfo


def _write_test_video(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    try:
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_convert_ctc_masks_to_box_tracks(tmp_path: Path) -> None:
    tra = tmp_path / "01_GT" / "TRA"
    tra.mkdir(parents=True)
    mask = np.zeros((20, 30), dtype=np.uint16)
    mask[4:10, 7:15] = 3
    assert cv2.imwrite(str(tra / "man_track000.tif"), mask)

    result = convert_multidomain_gt(
        source_format="ctc_masks_lineage",
        annotation_path=tmp_path,
        output_dir=tmp_path / "out",
    )

    assert result["categories"] == {"1": "cell"}
    assert result["sequences"][0]["track_count"] == 1
    line = Path(result["sequences"][0]["gt_path"]).read_text().strip()
    assert line.startswith("1,3,7.000000,4.000000,8.000000,6.000000")


def test_convert_ctc_uses_image_sequence_metadata(tmp_path: Path) -> None:
    tra = tmp_path / "dataset" / "01_GT" / "TRA"
    images = tmp_path / "dataset" / "01"
    tra.mkdir(parents=True)
    images.mkdir(parents=True)
    for frame_index in range(2):
        mask = np.zeros((20, 30), dtype=np.uint16)
        mask[4:10, 7:15] = 3
        assert cv2.imwrite(str(tra / f"man_track{frame_index:03d}.tif"), mask)
        assert cv2.imwrite(
            str(images / f"t{frame_index:03d}.tif"),
            np.zeros((40, 60), dtype=np.uint16),
        )

    result = convert_multidomain_gt(
        source_format="ctc_masks_lineage",
        annotation_path=tmp_path / "dataset",
        media_root=tmp_path / "dataset",
        output_dir=tmp_path / "out",
    )

    sequence = result["sequences"][0]
    assert sequence["frame_count"] == 2
    assert sequence["media_path"] == str(images.resolve())
    seqinfo = Path(sequence["seqinfo_path"]).read_text(encoding="utf-8")
    assert "imWidth=60" in seqinfo
    assert "imHeight=40" in seqinfo
    assert "imExt=.tif" in seqinfo


def test_ua_detrac_xml_conversion_preserves_vehicle_classes(tmp_path: Path) -> None:
    annotation = tmp_path / "MVI_1.xml"
    annotation.write_text(
        """<sequence name="MVI_1">
        <frame num="1"><target_list>
          <target id="7"><box left="1" top="2" width="30" height="20"/>
          <attribute vehicle_type="van" truncation_ratio="0.25"/></target>
        </target_list></frame>
        <frame num="2"><target_list>
          <target id="7"><box left="2" top="2" width="30" height="20"/>
          <attribute vehicle_type="van" truncation_ratio="0"/></target>
        </target_list></frame>
        </sequence>""",
        encoding="utf-8",
    )
    media = tmp_path / "media" / "MVI_1"
    media.mkdir(parents=True)
    for index in (1, 2):
        assert cv2.imwrite(
            str(media / f"img{index:05d}.jpg"),
            np.zeros((50, 80, 3), dtype=np.uint8),
        )

    result = convert_multidomain_gt(
        source_format="ua_detrac_xml",
        annotation_path=annotation,
        output_dir=tmp_path / "normalized",
        media_root=tmp_path / "media",
        media_fps=25.0,
        max_frames=1,
        overwrite=True,
    )

    assert result["annotation_count"] == 1
    assert result["categories"]["2"] == "van"
    gt = (tmp_path / "normalized" / "MVI_1" / "gt" / "gt.txt").read_text()
    assert gt.startswith(
        "1,7,1.000000,2.000000,30.000000,20.000000,1.000000,2,0.750000"
    )
    seqinfo = (tmp_path / "normalized" / "MVI_1" / "seqinfo.ini").read_text()
    assert "frameRate=25" in seqinfo

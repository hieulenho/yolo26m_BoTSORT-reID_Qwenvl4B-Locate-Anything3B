import collections
import json
import os

import datasets

_HOMEPAGE = "https://universe.roboflow.com/augmented-startups/football-player-detection-kucab"
_LICENSE = "CC BY 4.0"
_CITATION = """\
@misc{ football-player-detection-kucab_dataset,
    title = { Football-Player-Detection Dataset },
    type = { Open Source Dataset },
    author = { Augmented Startups },
    howpublished = { \\url{ https://universe.roboflow.com/augmented-startups/football-player-detection-kucab } },
    url = { https://universe.roboflow.com/augmented-startups/football-player-detection-kucab },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { nov },
    note = { visited on 2022-12-29 },
}
"""
_URLS = {
    "train": "https://huggingface.co/datasets/keremberke/football-object-detection/resolve/main/data/train.zip",
    "validation": "https://huggingface.co/datasets/keremberke/football-object-detection/resolve/main/data/valid.zip",
    "test": "https://huggingface.co/datasets/keremberke/football-object-detection/resolve/main/data/test.zip",
}

_CATEGORIES = ["player", "football"]
_ANNOTATION_FILENAME = "_annotations.coco.json"


class FOOTBALLOBJECTDETECTION(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=_CATEGORIES),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "folder_dir": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "folder_dir": data_files["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "folder_dir": data_files["test"],
                },
            ),
        ]

    def _generate_examples(self, folder_dir):
        def process_annot(annot, category_id_to_category):
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "category": category_id_to_category[annot["category_id"]],
            }

        image_id_to_image = {}
        idx = 0

        annotation_filepath = os.path.join(folder_dir, _ANNOTATION_FILENAME)
        with open(annotation_filepath) as f:
            annotations = json.load(f)
        category_id_to_category = {
            category["id"]: category["name"] for category in annotations["categories"]
        }
        image_id_to_annotations = collections.defaultdict(list)
        for annot in annotations["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        image_id_to_image = {annot["file_name"]: annot for annot in annotations["images"]}

        for filename in os.listdir(folder_dir):
            filepath = os.path.join(folder_dir, filename)
            if filename in image_id_to_image:
                image = image_id_to_image[filename]
                objects = [
                    process_annot(annot, category_id_to_category)
                    for annot in image_id_to_annotations[image["id"]]
                ]
                with open(filepath, "rb") as f:
                    image_bytes = f.read()
                yield (
                    idx,
                    {
                        "image_id": image["id"],
                        "image": {"path": filepath, "bytes": image_bytes},
                        "width": image["width"],
                        "height": image["height"],
                        "objects": objects,
                    },
                )
                idx += 1

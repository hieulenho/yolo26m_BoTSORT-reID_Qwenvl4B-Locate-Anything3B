from PIL import Image

from football_tracking.locate_tracking.grounding.locate_anything_backend import (
    LocateAnythingBackend,
    _resize_image_to_pixel_budget,
)


def test_locate_image_is_downscaled_to_pixel_budget() -> None:
    image = Image.new("RGB", (2560, 1440))

    resized = _resize_image_to_pixel_budget(image, 512 * 512)

    assert resized.size[0] * resized.size[1] <= 512 * 512
    assert resized.size[0] < image.size[0]
    assert abs(resized.size[0] / resized.size[1] - 16 / 9) < 0.01


def test_locate_inference_config_records_pixel_budget() -> None:
    backend = LocateAnythingBackend(image_max_pixels=131072)

    assert backend.inference_config()["image_max_pixels"] == 131072

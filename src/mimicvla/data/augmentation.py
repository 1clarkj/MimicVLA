"""Conservative training-time image augmentation using LeRobot's transforms."""


def appearance_augmentation_config():
    """Return LeRobot config for RGB appearance changes without pose changes."""
    from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig

    return ImageTransformsConfig(
        enable=True,
        max_num_transforms=2,
        random_order=False,
        tfs={
            "brightness": ImageTransformConfig(type="ColorJitter",
                                               kwargs={"brightness": (0.9, 1.1)}),
            "contrast": ImageTransformConfig(type="ColorJitter",
                                             kwargs={"contrast": (0.9, 1.1)}),
            "saturation": ImageTransformConfig(type="ColorJitter",
                                               kwargs={"saturation": (0.9, 1.1)}),
        },
    )

from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Compose(Transform):
    """Compose sequence of transforms.

    Args:
        transforms (iterable of Transform): Sequence of transforms.
    """

    def __init__(self, **transforms: Transform):
        self.transforms = transforms

    def __call__(self, x):
        """Apply transforms in sequence."""
        for transform in self.transforms:
            x = self.transforms[transform](x)
        return x

"""Exception types raised by the public API."""


class ImagejevError(Exception):
    """Base class for all imagejev errors."""


class QuestionSchemaError(ImagejevError, ValueError):
    """A question dict is malformed."""


class ImageLoadError(ImagejevError, ValueError):
    """An image could not be decoded."""

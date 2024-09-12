from __future__ import annotations

from functools import singledispatchmethod
from io import BytesIO

from altair.vegalite.v5.schema.core import TopLevelSpec as Altair
from matplotlib.figure import Figure as Matplotlib
from PIL.Image import Image as Pillow


class MediaItem:
    def __init__(self, media_bytes: bytes, media_encoding: str, media_type: str):
        self.media_bytes = media_bytes
        self.media_encoding = media_encoding
        self.media_type = media_type

    @singledispatchmethod
    @classmethod
    def factory(cls, media):
        raise NotImplementedError(f"Type '{type(media)}' is not yet supported")

    @factory.register(bytes)
    @classmethod
    def factory_bytes(
        cls,
        media: bytes,
        media_encoding: str = "utf-8",
        media_type: str = "application/octet-stream",
    ) -> MediaItem:
        return cls(
            media_bytes=media,
            media_encoding=media_encoding,
            media_type=media_type,
        )

    @factory.register(str)
    @classmethod
    def factory_str(cls, media: str, media_type: str = "text/html") -> MediaItem:
        media_bytes = media.encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type=media_type,
        )

    @factory.register(Altair)
    @classmethod
    def factory_altair(cls, media: Altair) -> MediaItem:
        media_bytes = media.to_json().encode("utf-8")

        return cls(
            media_bytes=media_bytes,
            media_encoding="utf-8",
            media_type="application/vnd.vega.v5+json",
        )

    @factory.register(Matplotlib)
    @classmethod
    def factory_matplotlib(cls, media: Matplotlib) -> MediaItem:
        with BytesIO() as stream:
            media.savefig(stream, format="svg")
            media_bytes = stream.getvalue()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/svg+xml",
            )

    @factory.register(Pillow)
    @classmethod
    def factory_pillow(cls, media: Pillow) -> MediaItem:
        with BytesIO() as stream:
            media.save(stream, format="png")
            media_bytes = stream.getvalue()

            return cls(
                media_bytes=media_bytes,
                media_encoding="utf-8",
                media_type="image/png",
            )

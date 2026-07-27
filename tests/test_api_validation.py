"""Tests for uploaded-image validation in the FastAPI prototype."""

import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from api import read_uploaded_image


def create_test_image_bytes() -> bytes:
    """Create a small valid PNG image in memory."""

    image = Image.new("RGB", (32, 32))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def create_upload_file(
    content: bytes,
    content_type: str,
    filename: str = "test-image.png",
) -> UploadFile:
    """Create an in-memory uploaded file for testing."""

    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


def test_valid_image_is_accepted() -> None:
    """A valid uploaded image should be decoded as RGB."""

    upload = create_upload_file(
        content=create_test_image_bytes(),
        content_type="image/png",
    )

    image = asyncio.run(read_uploaded_image(upload))

    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_invalid_image_content_is_rejected() -> None:
    """Invalid image bytes should produce a 400 response."""

    upload = create_upload_file(
        content=b"This is not a valid image.",
        content_type="image/png",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(read_uploaded_image(upload))

    assert error.value.status_code == 400


def test_non_image_media_type_is_rejected() -> None:
    """A non-image media type should produce a 415 response."""

    upload = create_upload_file(
        content=b"Example text file",
        content_type="text/plain",
        filename="example.txt",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(read_uploaded_image(upload))

    assert error.value.status_code == 415


def test_empty_upload_is_rejected() -> None:
    """An empty image upload should produce a 400 response."""

    upload = create_upload_file(
        content=b"",
        content_type="image/png",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(read_uploaded_image(upload))

    assert error.value.status_code == 400

"""Tests for vision/multimodal support in ChatMaritaca."""

from langchain_core.messages import HumanMessage

from langchain_maritaca.chat_models import (
    _convert_message_to_dict,
    _format_image_content,
)


class TestFormatImageContent:
    """Tests for _format_image_content function."""

    def test_string_content_passthrough(self) -> None:
        """Test that string content is passed through unchanged."""
        content = "Hello, world!"
        result = _format_image_content(content)
        assert result == "Hello, world!"

    def test_text_block(self) -> None:
        """Test text block formatting."""
        content = [{"type": "text", "text": "What is in this image?"}]
        result = _format_image_content(content)
        assert result == [{"type": "text", "text": "What is in this image?"}]

    def test_string_in_list_converted_to_text_block(self) -> None:
        """Test that plain strings in list are converted to text blocks."""
        content = ["Hello", "World"]
        result = _format_image_content(content)
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]

    def test_image_with_url(self) -> None:
        """Test image block with URL (LangChain standard format)."""
        content = [
            {"type": "text", "text": "O que há nesta imagem?"},
            {"type": "image", "url": "https://example.com/image.png"},
        ]
        result = _format_image_content(content)

        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "O que há nesta imagem?"}
        assert result[1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/image.png"},
        }

    def test_image_with_base64(self) -> None:
        """Test image block with base64 data."""
        content = [
            {"type": "text", "text": "Descreva esta imagem"},
            {
                "type": "image",
                "base64": "iVBORw0KGgoAAAANSUhEUg==",
                "mime_type": "image/png",
            },
        ]
        result = _format_image_content(content)

        assert len(result) == 2
        assert result[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgoAAAANSUhEUg==",
            },
        }

    def test_image_base64_default_mime_type(self) -> None:
        """Test that default mime_type is image/png when not specified."""
        content = [{"type": "image", "base64": "abc123"}]
        result = _format_image_content(content)

        assert result[0]["source"]["media_type"] == "image/png"

    def test_openai_image_url_format_with_url(self) -> None:
        """Test OpenAI image_url format with URL."""
        content = [
            {"type": "text", "text": "What's this?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.jpg"},
            },
        ]
        result = _format_image_content(content)

        assert len(result) == 2
        assert result[1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.jpg"},
        }

    def test_openai_image_url_format_with_data_uri(self) -> None:
        """Test OpenAI image_url format with data URI (base64)."""
        content = [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
            },
        ]
        result = _format_image_content(content)

        assert len(result) == 2
        assert result[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "/9j/4AAQSkZJRg==",
            },
        }

    def test_openai_image_url_string_format(self) -> None:
        """Test OpenAI image_url with string value instead of dict."""
        content = [
            {"type": "image_url", "image_url": "https://example.com/photo.png"},
        ]
        result = _format_image_content(content)

        assert result[0] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/photo.png"},
        }

    def test_mixed_content_types(self) -> None:
        """Test message with multiple text and image blocks."""
        content = [
            {"type": "text", "text": "Compare these images:"},
            {"type": "image", "url": "https://example.com/image1.png"},
            {"type": "text", "text": "vs"},
            {"type": "image", "url": "https://example.com/image2.png"},
        ]
        result = _format_image_content(content)

        assert len(result) == 4
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "text"
        assert result[3]["type"] == "image"

    def test_unknown_block_type_passthrough(self) -> None:
        """Test that unknown block types are passed through."""
        content = [{"type": "custom", "data": "value"}]
        result = _format_image_content(content)

        assert result == [{"type": "custom", "data": "value"}]


class TestConvertMessageToDictVision:
    """Tests for vision support in _convert_message_to_dict."""

    def test_human_message_with_image_url(self) -> None:
        """Test HumanMessage with image URL."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "O que você vê?"},
                {"type": "image", "url": "https://example.com/cat.jpg"},
            ]
        )
        result = _convert_message_to_dict(message)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "O que você vê?"}
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["type"] == "url"

    def test_human_message_with_base64_image(self) -> None:
        """Test HumanMessage with base64 encoded image."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analise esta imagem"},
                {
                    "type": "image",
                    "base64": "iVBORw0KGgoAAAANSUhEUg==",
                    "mime_type": "image/png",
                },
            ]
        )
        result = _convert_message_to_dict(message)

        assert result["role"] == "user"
        assert result["content"][1]["source"]["type"] == "base64"
        assert result["content"][1]["source"]["media_type"] == "image/png"

    def test_human_message_openai_format(self) -> None:
        """Test HumanMessage with OpenAI image_url format."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ]
        )
        result = _convert_message_to_dict(message)

        assert result["role"] == "user"
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["url"] == "https://example.com/img.png"

    def test_human_message_string_content_unchanged(self) -> None:
        """Test that string content in HumanMessage works as before."""
        message = HumanMessage(content="Hello!")
        result = _convert_message_to_dict(message)

        assert result == {"role": "user", "content": "Hello!"}

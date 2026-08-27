import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.utils.text import (
    normalize_text,
    compare_names,
    compute_string_similarity,
    extract_candidate_name_from_lines,
)
from app.services.ocr_service import OCRService
from app.core.exceptions import OCRFailedException


class TestTextAndNameMatching:
    def test_normalize_text(self):
        assert normalize_text("  JOHN   DOE  ") == "john doe"
        assert normalize_text("John-Doe, Jr.") == "john doe jr"
        assert normalize_text("Éléonore Smith") == "eleonore smith"

    def test_exact_name_match(self):
        match, score = compare_names("SUNAV SHARMA", "SUNAV SHARMA")
        assert match is True
        assert score == 1.0

    def test_case_insensitive_name_match(self):
        match, score = compare_names("sunav sharma", "SUNAV SHARMA")
        assert match is True
        assert score == 1.0

    def test_whitespace_normalized_name_match(self):
        match, score = compare_names("  SUNAV    SHARMA \n ", "SUNAV SHARMA")
        assert match is True
        assert score == 1.0

    def test_reordered_tokens_name_match(self):
        match, score = compare_names("SHARMA SUNAV", "SUNAV SHARMA")
        assert match is True
        assert score == 1.0

    def test_name_mismatch(self):
        match, score = compare_names("SUNAV SHARMA", "JANE DOE")
        assert match is False
        assert score < 0.65

    def test_extract_candidate_name_from_labeled_lines(self):
        lines = [
            "DRIVING LICENSE",
            "LIC NO: DL-99882233",
            "NAME: JOHN FITZGERALD DOE",
            "DOB: 1990-01-01",
            "ADDRESS: 123 MAIN ST",
        ]
        name = extract_candidate_name_from_lines(lines)
        assert name == "JOHN FITZGERALD DOE"

    def test_extract_candidate_name_multiline(self):
        lines = [
            "REPUBLIC OF TEST",
            "IDENTITY CARD",
            "NAME",
            "ALICE SMITH",
            "SEX: F",
        ]
        name = extract_candidate_name_from_lines(lines)
        assert name == "ALICE SMITH"

    def test_extract_candidate_name_unlabeled(self):
        lines = [
            "REPUBLIC OF TEST",
            "ROBERT BRUCE BANNER",
            "1985-05-12",
        ]
        name = extract_candidate_name_from_lines(lines)
        assert name == "ROBERT BRUCE BANNER"

    def test_failed_ocr_no_name_found(self):
        lines = [
            "123456789",
            "2025-01-01",
            "--- ... ---",
        ]
        name = extract_candidate_name_from_lines(lines)
        assert name is None


class TestOCRService:
    def test_ocr_service_empty_image_error(self):
        service = OCRService()
        with pytest.raises(OCRFailedException):
            service.extract_text(np.zeros((0, 0, 3), dtype=np.uint8))

    @patch("paddleocr.PaddleOCR")
    def test_ocr_service_extraction(self, mock_ocr_cls):
        mock_instance = MagicMock()
        mock_instance.ocr.return_value = [[
            [
                [[10, 10], [100, 10], [100, 30], [10, 30]],
                ("NAME: JANE DOE", 0.98),
            ],
            [
                [[10, 40], [100, 40], [100, 60], [10, 60]],
                ("DOB: 1995-10-15", 0.95),
            ],
        ]]
        mock_ocr_cls.return_value = mock_instance

        service = OCRService()
        service._ocr = mock_instance
        service._initialized = True

        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = service.extract_text(dummy_img)

        assert result["success"] is True
        assert len(result["lines"]) == 2
        assert result["extracted_name"] == "JANE DOE"

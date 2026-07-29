from dataclasses import replace

import numpy as np
import pytest

from digit_sdk.depth_contract import (
    make_depth_result,
    validate_depth_raw_mm,
    validate_depth_result,
)


def test_make_depth_result_preserves_raw_and_applies_cutoff():
    source = np.array([[0.0, 0.05], [0.1, 0.2]], dtype=np.float32)

    result = make_depth_result(source, depth_cutoff_mm=0.1)

    np.testing.assert_array_equal(result.depth_raw_mm, source)
    np.testing.assert_array_equal(
        result.mask, np.array([[False, False], [True, True]])
    )
    np.testing.assert_array_equal(
        result.depth_output_mm,
        np.array([[0.0, 0.0], [0.1, 0.2]], dtype=np.float32),
    )
    assert result.depth_raw_mm is not source
    source.fill(9)
    assert result.depth_raw_mm[0, 0] == 0


def test_zero_cutoff_disables_suppression():
    raw = np.array([[0.0, 0.05]], dtype=np.float32)

    result = make_depth_result(raw, depth_cutoff_mm=0.0)

    assert np.all(result.mask)
    np.testing.assert_array_equal(result.depth_output_mm, raw)


@pytest.mark.parametrize(
    "raw, error",
    [
        (np.zeros((2, 2), dtype=np.float64), "dtype float32"),
        (np.zeros((2, 2, 1), dtype=np.float32), "HxW"),
        (np.empty((0, 2), dtype=np.float32), "non-empty"),
        (np.array([[np.nan]], dtype=np.float32), "finite"),
        (np.array([[-0.01]], dtype=np.float32), "non-negative"),
    ],
)
def test_raw_depth_validation_rejects_contract_violations(raw, error):
    with pytest.raises(ValueError, match=error):
        validate_depth_raw_mm(raw)


def test_raw_depth_validation_checks_expected_shape():
    raw = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="expected shape"):
        validate_depth_raw_mm(raw, expected_shape=(3, 2))


@pytest.mark.parametrize("cutoff", [-0.1, np.inf, np.nan])
def test_make_depth_result_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match="finite and non-negative"):
        make_depth_result(np.zeros((2, 2), dtype=np.float32),
                          depth_cutoff_mm=cutoff)


def test_complete_result_validation_detects_modified_output():
    result = make_depth_result(
        np.array([[0.05, 0.2]], dtype=np.float32),
        depth_cutoff_mm=0.1,
    )
    invalid_output = result.depth_output_mm.copy()
    invalid_output[0, 0] = 0.05

    with pytest.raises(ValueError, match="masked depth_raw_mm"):
        validate_depth_result(
            replace(result, depth_output_mm=invalid_output),
            depth_cutoff_mm=0.1,
        )


def test_complete_result_validation_accepts_matching_result():
    result = make_depth_result(
        np.array([[0.05, 0.2]], dtype=np.float32),
        depth_cutoff_mm=0.1,
    )

    validate_depth_result(result, depth_cutoff_mm=0.1)

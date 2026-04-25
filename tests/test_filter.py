import pytest
from kickroll_beatmap_generator.filter import validate_sections, filter_timestamps


def test_validate_section_start_negative():
    with pytest.raises(ValueError):
        validate_sections([(-1.0, 5.0)], duration_sec=10.0)


def test_validate_section_end_exceeds_duration():
    with pytest.raises(ValueError):
        validate_sections([(0.0, 15.0)], duration_sec=10.0)


def test_validate_section_start_after_end():
    with pytest.raises(ValueError):
        validate_sections([(5.0, 3.0)], duration_sec=10.0)


def test_validate_section_start_equals_end():
    with pytest.raises(ValueError):
        validate_sections([(5.0, 5.0)], duration_sec=10.0)


def test_filter_includes_boundaries():
    result = filter_timestamps([2000.0, 3000.0], [(2.0, 3.0)])
    assert 2000.0 in result
    assert 3000.0 in result


def test_filter_excludes_outside():
    result = filter_timestamps([1999.9, 3000.1], [(2.0, 3.0)])
    assert result == []


def test_filter_overlapping_sections_no_duplicate():
    result = filter_timestamps([2500.0], [(2.0, 3.0), (2.4, 2.6)])
    assert result == [2500.0]


def test_filter_result_sorted():
    result = filter_timestamps([3000.0, 1000.0, 2000.0], [(0.0, 4.0)])
    assert result == [1000.0, 2000.0, 3000.0]


def test_filter_empty_result_valid():
    result = filter_timestamps([5000.0, 6000.0], [(0.0, 2.0)])
    assert result == []

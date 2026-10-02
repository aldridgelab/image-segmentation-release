from __future__ import annotations

from pathlib import Path

from deciphaer_image_segmentation import skiplist


def test_load_returns_empty_set_when_file_missing(tmp_path: Path) -> None:
    assert skiplist.load(tmp_path / "skip.txt") == set()


def test_load_strips_comments_and_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "skip.txt"
    p.write_text("# header\n\nimage_a\nimage_b  # inline\n# trailing\n", encoding="utf-8")
    assert skiplist.load(p) == {"image_a", "image_b"}


def test_matches_against_name_stem_and_resolved_path(tmp_path: Path) -> None:
    src = tmp_path / "subdir" / "img_001.tif"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"")
    assert skiplist.matches({"img_001.tif"}, src)
    assert skiplist.matches({"img_001"}, src)
    assert skiplist.matches({str(src.resolve())}, src)
    assert not skiplist.matches({"other"}, src)
    assert not skiplist.matches(set(), src)


def test_append_writes_header_and_dedupes(tmp_path: Path) -> None:
    p = tmp_path / "skip.txt"
    added_first = skiplist.append(p, ["a", "b", ""], comment="manual")
    assert added_first == ["a", "b"]
    body = p.read_text(encoding="utf-8")
    assert body.startswith(skiplist.HEADER)
    assert "manual" in body

    # Re-appending the same tokens is a no-op; new tokens get added.
    added_second = skiplist.append(p, ["a", "c"])
    assert added_second == ["c"]
    assert skiplist.load(p) == {"a", "b", "c"}


def test_append_returns_empty_when_all_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "skip.txt"
    skiplist.append(p, ["a"])
    assert skiplist.append(p, ["a"]) == []

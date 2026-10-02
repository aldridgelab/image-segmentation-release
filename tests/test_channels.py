from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from deciphaer_image_segmentation.channels import (
    FLUOR_COMMON_SUFFIXES,
    HadaPlugin,
    INTENSITY_SUFFIXES,
    KNOWN_CHANNELS,
    MORPHOLOGY_FEATURE_COLUMNS,
    PROFILE_SUFFIXES,
    PluginContext,
    PunctaPlugin,
    discover_channels_in_dir,
    discover_sibling_channels,
    feature_columns_for_channels,
    parse_channel_token,
    plugins_for_channel,
)


def test_unknown_channel_gets_common_features_only() -> None:
    """A channel not matching any plugin (e.g. Fluor1) should get only the
    common intensity/profile/fluor-common families — no plugin extras."""
    cols = feature_columns_for_channels(
        ("Phase", "Fluor1"), phase_channel="Phase",
    )

    # Common families per channel are emitted.
    assert "intensity_Fluor1_mean" in cols
    assert "profile_Fluor1_mean" in cols
    assert "Fluor1_bg_mean" in cols
    assert "Fluor1_midline_mean" in cols

    # Plugin extras stay attached to their respective channels only.
    assert not any(c.startswith("Fluor1_poleA") for c in cols)
    assert not any(c.startswith("Fluor1_puncta_") for c in cols)


def test_af405_channel_gets_hada_extras() -> None:
    cols = feature_columns_for_channels(
        ("Phase", "AF405"), phase_channel="Phase",
    )
    for suf in HadaPlugin.extra_suffixes:
        assert f"AF405_{suf}" in cols


def test_bod493_channel_gets_puncta_extras() -> None:
    cols = feature_columns_for_channels(
        ("Phase", "Bod493"), phase_channel="Phase",
    )
    for suf in PunctaPlugin.extra_suffixes:
        assert f"Bod493_{suf}" in cols


def test_plugin_matching_is_substring() -> None:
    """A channel whose name contains AF405 (e.g. AF405_lib1) still gets HADA."""
    plugins = plugins_for_channel("AF405_lib1")
    assert any(isinstance(p, HadaPlugin) for p in plugins)


def test_phase_only_run_has_no_fluor_columns() -> None:
    cols = feature_columns_for_channels(("Phase",), phase_channel="Phase")
    assert "intensity_Phase_mean" in cols
    assert "profile_Phase_mean" in cols
    # Phase doesn't go through the fluorescence pass.
    assert not any(c.startswith("Phase_bg_") for c in cols)


def test_feature_column_count_matches_legacy_reference() -> None:
    """Resolving the legacy channel set (Phase, AF405, Bod493) yields the
    same 176 columns as the old hardcoded DEFAULT_FEATURE_COLUMNS."""
    cols = feature_columns_for_channels(
        ("Phase", "AF405", "Bod493"), phase_channel="Phase",
    )
    expected = (
        len(MORPHOLOGY_FEATURE_COLUMNS)
        + 3 * len(INTENSITY_SUFFIXES)          # Phase + AF405 + Bod493 intensity
        + 3 * len(PROFILE_SUFFIXES)            # Phase + AF405 + Bod493 profile
        + 2 * len(FLUOR_COMMON_SUFFIXES)       # AF405 + Bod493 fluor-common
        + len(HadaPlugin.extra_suffixes)        # AF405 HADA extras
        + len(PunctaPlugin.extra_suffixes)      # Bod493 puncta extras
    )
    assert len(cols) == expected == 176


def test_discover_channels_finds_siblings(tmp_path: Path) -> None:
    img = np.zeros((4, 4), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "site01_Phase.tif", img)
    tifffile.imwrite(tmp_path / "site01_AF405.tif", img)
    tifffile.imwrite(tmp_path / "site01_Bod493.tif", img)
    tifffile.imwrite(tmp_path / "site01_Phase_mask.tif", img)  # should be skipped

    found = discover_channels_in_dir(tmp_path, phase_channel="Phase")
    assert found[0] == "Phase"
    assert set(found[1:]) == {"AF405", "Bod493"}
    assert "Phase_mask" not in found


def test_discover_channels_empty_dir(tmp_path: Path) -> None:
    found = discover_channels_in_dir(tmp_path, phase_channel="Phase")
    assert found == ("Phase",)


def test_parse_channel_token_strips_trailing_tag() -> None:
    """Zen-style exports like ``..._Cy5_ORG.tif`` should still resolve to ``Cy5``.

    The trailing ``_ORG`` is unrelated to the channel name and would otherwise
    poison every column header. The registry-based parser anchors on
    ``_Cy5`` followed by a separator (``_`` here) so the channel identity
    survives the export-tool's suffix.
    """
    stem = "mjm_2026_Mabs_OGDA_s01m01_Cy5_ORG"
    channel, base = parse_channel_token(stem)
    assert channel == "Cy5"
    assert base == "mjm_2026_Mabs_OGDA_s01m01"


def test_parse_channel_token_handles_spaced_qualifier() -> None:
    """A Phase channel labelled ``Phase PH3`` in the filename should still parse.

    The space after ``_Phase`` counts as a separator, so the channel is
    ``Phase`` and the qualifier + any trailing tag are stripped from the base.
    """
    stem = "site01_Phase PH3_ORG"
    channel, base = parse_channel_token(stem)
    assert channel == "Phase"
    assert base == "site01"


def test_parse_channel_token_ignores_project_name_substring() -> None:
    """A channel-shaped substring inside the project name must not false-match.

    ``sCy5DA-01`` contains ``Cy5`` but is preceded by ``s``, not ``_``, so the
    parser rejects it. Only the genuine trailing ``_Cy5`` token wins.
    """
    stem = "mjm_2026_Mabs_OGDA_sCy5DA-01_s01m01_Phase"
    channel, base = parse_channel_token(stem)
    assert channel == "Phase"
    assert base == "mjm_2026_Mabs_OGDA_sCy5DA-01_s01m01"


def test_parse_channel_token_falls_back_to_strict_suffix() -> None:
    """When no registry channel hits, the legacy ``_<phase>`` suffix logic kicks in.

    Keeps old naming working even if the user adds a custom phase token
    that isn't in :data:`KNOWN_CHANNELS`.
    """
    channel, base = parse_channel_token(
        "site42_CustomPhase",
        known_channels=("Cy5",),  # Phase intentionally omitted.
        phase_channel="CustomPhase",
    )
    assert channel == "CustomPhase"
    assert base == "site42"


def test_discover_sibling_channels_handles_ORG_suffix(tmp_path: Path) -> None:
    """The Mabs-pulse-chase naming (``..._Cy5_ORG.tif``) should now discover Cy5+OG514."""
    img = np.zeros((4, 4), dtype=np.uint16)
    base = "mjm_2026_Mabs_OGDA_s01m01"
    phase_path = tmp_path / f"{base}_Phase PH3_ORG.tif"
    cy5_path = tmp_path / f"{base}_Cy5_ORG.tif"
    og_path = tmp_path / f"{base}_OG514_ORG.tif"
    tifffile.imwrite(phase_path, img)
    tifffile.imwrite(cy5_path, img)
    tifffile.imwrite(og_path, img)

    siblings = discover_sibling_channels(phase_path, phase_channel="Phase")
    assert set(siblings) == {"Cy5", "OG514"}
    assert siblings["Cy5"].name == cy5_path.name
    assert siblings["OG514"].name == og_path.name


def test_discover_channels_handles_ORG_suffix(tmp_path: Path) -> None:
    img = np.zeros((4, 4), dtype=np.uint16)
    base = "mjm_2026_Mabs_OGDA_s01m01"
    tifffile.imwrite(tmp_path / f"{base}_Phase PH3_ORG.tif", img)
    tifffile.imwrite(tmp_path / f"{base}_Cy5_ORG.tif", img)
    tifffile.imwrite(tmp_path / f"{base}_OG514_ORG.tif", img)

    found = discover_channels_in_dir(tmp_path, phase_channel="Phase")
    assert found[0] == "Phase"
    assert set(found[1:]) == {"Cy5", "OG514"}


def test_hada_plugin_empty_features_keys() -> None:
    """A plugin's empty_features dict must cover all the keys it advertises."""
    plugin = HadaPlugin()
    empty = plugin.empty_features("AF405")
    assert set(empty) == {f"AF405_{s}" for s in plugin.extra_suffixes}
    for v in empty.values():
        assert np.isnan(v)


def test_hada_plugin_extract_on_synthetic_profile() -> None:
    """A clean polar profile should produce sensible HADA scalars."""
    plugin = HadaPlugin()

    # Synthesize a profile bright at one pole, dim at the other.
    profile = np.concatenate([
        np.linspace(10.0, 5.0, 50),
        np.linspace(5.0, 1.0, 50),
    ])

    class _Cfg:
        pole_window = 0.10
        center_window = (0.45, 0.55)

    ctx = PluginContext(
        channel="AF405",
        image=np.zeros((4, 4)),
        binary_mask=np.zeros((4, 4), dtype=np.uint8),
        cell_pixels=np.zeros(0),
        midline=None,
        profile=profile,
        core_mask=None,
        shell_mask=None,
        bg_value=0.0,
        fluor_cfg=_Cfg(),
        puncta_records_out=[],
    )
    feats = plugin.extract(ctx)
    assert feats["AF405_poleA"] > feats["AF405_poleB"]
    assert feats["AF405_polar_asym"] > 0
    assert "AF405_state_code" in feats

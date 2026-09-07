import json

import pytest

from treesupport.ui_settings import (SUPPORT_FIELDS, INFILL_FIELDS, PRESETS,
                                    configuration, load_settings, save_settings)


def defaults():
    return {key: value for key, (_, value) in (SUPPORT_FIELDS | INFILL_FIELDS).items()}


def test_settings_roundtrip_and_corrupt_input(tmp_path):
    path = tmp_path / "settings.json"
    values = defaults()
    values["cell_size"] = "12"
    save_settings(path, values, "infill", "diamond")
    loaded = load_settings(path)
    assert loaded["mode"] == "infill"
    assert loaded["pattern"] == "diamond"
    assert configuration(loaded["values"], "infill", loaded["pattern"]).cell_size == 12
    loaded["values"]["pitch"] = "nan"
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_validate_for_both_modes(preset):
    values = defaults() | PRESETS[preset]
    configuration(values, "support")
    configuration(values, "infill")

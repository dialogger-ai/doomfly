"""Guard the time-separated, dose-matched conditioning control."""

from asteroids.neural import GAME_HZ, NEURAL_STEPS_PER_SECOND
from asteroids.policy_temporal_cue_conditioning import (
    NEUTRAL_TICKS, PULSE_STEPS, TRAIN_CUE_TICKS, pulse_steps, stimulus_pair,
)


def test_equal_dose_with_three_seconds_of_neutral_before_delayed_pulse():
    paired = pulse_steps("paired")
    delayed = pulse_steps("unpaired")
    assert paired[1] - paired[0] == delayed[1] - delayed[0] == PULSE_STEPS
    assert paired[1] <= round(TRAIN_CUE_TICKS * NEURAL_STEPS_PER_SECOND / GAME_HZ)
    assert delayed[0] - round(TRAIN_CUE_TICKS * NEURAL_STEPS_PER_SECOND / GAME_HZ) == 3 * NEURAL_STEPS_PER_SECOND
    assert delayed[1] <= round((TRAIN_CUE_TICKS + NEUTRAL_TICKS) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
    assert pulse_steps("withheld") is None


def test_cue_pair_is_selected_by_metadata_not_list_order():
    rows = [dict(pair_id="pair", direction=2, target_radius=1., radial_speed=2.,
                 condition=name, label=label, seed=7)
            for name, label in (("recovery_outward", 1), ("safe_inward", 0))]
    assert [cue.label for cue in stimulus_pair(rows)] == [0, 1]

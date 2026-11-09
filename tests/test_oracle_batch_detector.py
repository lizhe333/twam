import random

import numpy as np

from contact_belief_wam.oracle_batch_detector import PrivilegedStallDetector


def scenario(*, sustained=True, moving=False, command=True):
    detector = PrivilegedStallDetector([0, 0, -1])
    for t in range(21):
        touch = np.zeros((2, 4, 4, 3), np.uint8)
        if (sustained and t >= 6) or (not sustained and t == 10):
            touch[:] = 100
        detector.observe(touch=touch, object_position=[0, 0, -t*.0001 if moving else 0],
                         observed_fk=[0, 0, 0], command_fk=[0, 0, -.0001 if command else 0] if t else None,
                         distance=.0001 if t >= 5 else .01)
    return detector


def test_stall_confirmed_after_one_chunk_and_causal_evidence():
    d = scenario()
    assert d.first_confirmed == 10
    assert d.candidate["t_contact"] == 5
    assert d.candidate["t_tactile"] == 6
    assert d.candidate["input_tactile_indices"] == [4, 6, 8, 10]
    assert d.events[0]["object_progress_mm"] == 0


def test_insertion_and_transient_touch_do_not_trigger():
    assert scenario(moving=True).first_confirmed is None
    assert scenario(sustained=False).first_confirmed is None
    assert scenario(command=False).first_confirmed is None


def test_real_progress_clears_old_failure_even_without_new_downward_command():
    d = scenario()
    for t in range(21, 31):
        d.observe(touch=np.full((2, 4, 4, 3), 100, np.uint8), object_position=[0, 0, -(t-20)*.00003],
                  observed_fk=[0, 0, 0], command_fk=[0, 0, 0], distance=.0001)
    assert d.first_confirmed is None
    assert d.events[-1]["status"] == "progress_resumed"


def test_slow_creep_does_not_erase_early_contact_snapshot():
    # Both missed HDMI failures were confirmed at 60, then cleared by only
    # 0.046/0.078 mm of progress while contact and tactile evidence persisted.
    for progress_mm, command_mm in ((.045972, .138059), (.077712, .191241), (.06, .21)):
        d = scenario()
        for t in range(21, 31):
            d.observe(touch=np.full((2, 4, 4, 3), 100, np.uint8),
                      object_position=[0, 0, -(t-20)*progress_mm/10000],
                      observed_fk=[0, 0, 0], command_fk=[0, 0, -command_mm/10000], distance=.00048)
        assert d.first_confirmed == 10
        assert d.events[-1]["status"] == "persistent_candidate"
        assert d.events[-1]["resume_required_mm"] == .2


def test_recording_does_not_consume_rng():
    random.seed(17)
    np.random.seed(21)
    state, npstate = random.getstate(), np.random.get_state()
    scenario()
    assert state == random.getstate()
    after = np.random.get_state()
    assert npstate[0] == after[0]
    np.testing.assert_array_equal(npstate[1], after[1])
    assert npstate[2:] == after[2:]


def test_two_chunks_when_contact_evidence_arrives_late():
    d = PrivilegedStallDetector([0, 0, -1])
    for t in range(21):
        d.observe(touch=np.full((2, 4, 4, 3), 100 if t >= 10 else 0, np.uint8),
                  object_position=[0, 0, 0], observed_fk=[0, 0, 0],
                  command_fk=[0, 0, -.0001] if t else None, distance=.0001 if t >= 9 else .01)
    assert d.events[0]["status"] == "unconfirmed"
    assert d.first_confirmed == 20

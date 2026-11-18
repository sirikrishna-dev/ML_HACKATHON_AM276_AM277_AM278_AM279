"""HMM/Viterbi decoder interface. P3 owns implementation."""


def decode(cands_df, trip, cfg):
    """Return columns obs_idx, arc_id, break_flag; must not read trip.route."""
    raise NotImplementedError("P3: implement HMM/Viterbi decoder")

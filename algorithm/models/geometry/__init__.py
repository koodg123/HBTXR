"""models.geometry — pupil-state representation and geometric codecs.

Shared by all modalities: the canonical state s=(x,y,a,b,theta) with a>=b and
theta in [0,pi) (state.py), the box-to-state decoder g() and the residual update
s_cand=Pi(z+ds) (codec.py).
"""
from models.geometry.codec import apply_residual, box_to_state, state_to_box
from models.geometry.state import STATE_DIM, canonicalize, wrap_pi

__all__ = ["STATE_DIM", "canonicalize", "wrap_pi", "box_to_state", "state_to_box", "apply_residual"]

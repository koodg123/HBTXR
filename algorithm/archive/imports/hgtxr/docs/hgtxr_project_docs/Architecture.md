# HGTXR Architecture

HGTXR uses a hybrid frame/event pipeline:

1. Frame patch embedding.
2. Event polarity patch embedding.
3. Shared HG-PIPE-style token backbone.
4. Search and event heads for robust relocalization.
5. Previous-state-conditioned fusion.
6. Track head for low-latency update.
7. Runtime FSM for search/track/hold decisions.


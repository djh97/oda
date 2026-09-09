"""Compatibility import for the versioned synthetic reference policy."""

from .policy import (  # noqa: F401
    DEFAULT_PROTOCOL_PATH,
    PolicyInputError,
    abo_compatible,
    hla_overlap,
    load_protocol,
    normalize_blood_group,
    rank_recipients_baseline,
    selectable_order,
)

"""Harrington LMI branding — delegates to harrington_common.theme.

All public names are preserved so existing page imports work unchanged.
"""
from __future__ import annotations

from harrington_common.theme import (  # noqa: F401
    BRAND,
    apply_brand_css,
    aw_panel,
    esc,
    st_svg,
)

# Alias so existing pages importing lmi_panel keep working
lmi_panel = aw_panel

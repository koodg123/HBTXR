from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_prefetchall4_300_contract_audit as audit_tool  # noqa: E402


class PrefetchAll4ContractAuditTests(unittest.TestCase):
    def test_contract_audit_has_required_core_checks(self) -> None:
        audit = audit_tool.build_audit()
        names = {check["name"] for check in audit["checks"]}

        self.assertGreaterEqual(audit["summary"]["checks_total"], 15)
        self.assertIn("runtime_scheduler_present", names)
        self.assertIn("frame_event_conv_onchip_rom_range", names)
        self.assertIn("head_onchip_rom_range", names)
        self.assertIn("track_transformer_onchip_rom_range", names)
        self.assertIn("three_nonlinear_roms_bound", names)
        self.assertIn("search_dispatcher_prefetch_storage_uram", names)
        self.assertIn("search_dispatcher_prefetch_all4", names)
        self.assertIn("full_transformer_modules_generated", names)

    def test_contract_audit_current_profile_passes(self) -> None:
        audit = audit_tool.build_audit()

        self.assertEqual(audit["status"], "pass", [check for check in audit["checks"] if check["status"] != "pass"])


if __name__ == "__main__":
    unittest.main()

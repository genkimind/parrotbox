import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parrotbox import persona


class PersonaTests(unittest.TestCase):
    def test_local_persona_overrides_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "persona.example.md"
            local = root / "persona.local.md"
            example.write_text("example", encoding="utf-8")
            local.write_text("private persona", encoding="utf-8")

            with (
                patch.object(persona, "PERSONA_FILE", example),
                patch.object(persona, "LOCAL_PERSONA_FILE", local),
            ):
                self.assertEqual(persona.build_system_prompt(), "private persona")

    def test_scenario_and_meeting_instructions_are_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "persona.example.md"
            example.write_text("stable persona", encoding="utf-8")

            with (
                patch.object(persona, "PERSONA_FILE", example),
                patch.object(persona, "LOCAL_PERSONA_FILE", root / "missing.md"),
            ):
                prompt = persona.build_system_prompt("daily standup", meeting=True)

        self.assertTrue(prompt.startswith("stable persona"))
        self.assertIn("daily standup", prompt)
        self.assertIn("## Meeting mode", prompt)


if __name__ == "__main__":
    unittest.main()

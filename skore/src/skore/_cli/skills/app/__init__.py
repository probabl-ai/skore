"""Textual applications backing the interactive ``skore`` CLI commands."""

from skore._cli.skills.app._find import ProbablSkillsFinder
from skore._cli.skills.app._install import ProbablSkillsInstaller

__all__ = ["ProbablSkillsFinder", "ProbablSkillsInstaller"]

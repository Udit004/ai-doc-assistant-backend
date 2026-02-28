"""Abstract base class that every agent tool must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Minimal interface for agent tools.

    Each tool encapsulates one capability (retrieval, answer generation, …)
    so the agent can swap implementations without changing the loop.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and traces."""

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        Execute the tool synchronously.

        Parameters
        ----------
        query : str
            The input string for this tool (a sub-question or search query).
        **kwargs
            Tool-specific extra arguments (e.g. ``db``, ``user_id``, ``context``).

        Returns
        -------
        str
            The tool's output as a plain string.
        """

"""Startup environment diagnostics for the smart capture runtime."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aoe4.smart_controller_config import SmartControllerConfig


REQUIRED_MODULES = ("PyQt5", "mss", "rapidocr", "onnxruntime")


@dataclass(frozen=True)
class DependencyStatus:
    module: str
    available: bool


@dataclass(frozen=True)
class EnvironmentReport:
    dependencies: tuple[DependencyStatus, ...]
    config_available: bool
    config_error: str | None

    @property
    def ready(self) -> bool:
        return self.config_available and all(
            dep.available for dep in self.dependencies
        )

    @property
    def messages_zh(self) -> tuple[str, ...]:
        lines: list[str] = []
        missing = [
            dep.module for dep in self.dependencies if not dep.available
        ]
        if missing:
            lines.append("缺少依赖模块: " + ", ".join(missing))
        if self.config_available:
            lines.append("配置文件有效")
        else:
            detail = self.config_error or "未知错误"
            lines.append("配置文件无效: " + detail)
        if self.ready:
            lines.insert(0, "环境已就绪")
        return tuple(lines)


class EnvironmentValidator:
    def __init__(
        self,
        module_finder: Callable[[str], object] = importlib.util.find_spec,
    ) -> None:
        if not callable(module_finder):
            raise TypeError("module_finder must be callable")
        self._module_finder = module_finder

    def _check_dependency(self, name: str) -> DependencyStatus:
        try:
            found = self._module_finder(name)
        except Exception:
            return DependencyStatus(module=name, available=False)
        return DependencyStatus(module=name, available=found is not None)

    def validate(self, config_path: str | Path) -> EnvironmentReport:
        dependencies = tuple(
            self._check_dependency(name) for name in REQUIRED_MODULES
        )
        path = Path(config_path)
        if not path.is_file():
            return EnvironmentReport(
                dependencies=dependencies,
                config_available=False,
                config_error="配置文件不存在或不是文件",
            )
        try:
            SmartControllerConfig.from_file(path)
        except Exception as exc:
            return EnvironmentReport(
                dependencies=dependencies,
                config_available=False,
                config_error=f"{type(exc).__name__}: {exc}",
            )
        return EnvironmentReport(
            dependencies=dependencies,
            config_available=True,
            config_error=None,
        )

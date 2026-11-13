# /src/app/services/testing/quality_config.py
"""
Quality Configuration Management Module
Handles tool registry, configuration, and settings for quality analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration for a quality tool.

    Attributes:
        name (str): Tool name.
        cmd (List[str]): Command to execute the tool.
        fix_cmd (Optional[List[str]]): Command to fix issues (if supported).
        description (str): Tool description.
        supports_fix (bool): Whether tool supports automatic fixes.
        output_format (str): Output format (json, text, diff, xml).
        timeout (int): Execution timeout in seconds.
        enabled (bool): Whether tool is enabled.
        priority (int): Tool execution priority.
        config_file (Optional[str]): Path to tool configuration file.
        extra_args (List[str]): Additional command line arguments.
    """

    name: str
    cmd: list[str]
    fix_cmd: list[str] | None = None
    description: str = ""
    supports_fix: bool = False
    output_format: str = "text"
    timeout: int = 60
    enabled: bool = True
    priority: int = 1
    config_file: str | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class LanguageConfig:
    """Configuration for a programming language.

    Attributes:
        name (str): Language name.
        extensions (List[str]): File extensions for this language.
        tools (Dict[str, ToolConfig]): Available tools for this language.
        default_tools (List[str]): Default tools to run.
        container_image (str): Docker image for analysis.
        install_commands (List[str]): Commands to install tools in container.
    """

    name: str
    extensions: list[str]
    tools: dict[str, ToolConfig] = field(default_factory=dict)
    default_tools: list[str] = field(default_factory=list)
    container_image: str = "code-graph-test:latest"
    install_commands: list[str] = field(default_factory=list)


@dataclass
class QualityAnalysisConfig:
    """Configuration for quality analysis behavior.

    Attributes:
        max_file_size_mb (int): Maximum file size to analyze.
        timeout_seconds (int): Global timeout for analysis.
        parallel_tools (bool): Run tools in parallel.
        cache_results (bool): Cache analysis results.
        cache_ttl_hours (int): Cache TTL in hours.
        auto_fix (bool): Apply automatic fixes.
        fail_on_error (bool): Fail analysis on tool errors.
        severity_threshold (str): Minimum severity to report.
    """

    max_file_size_mb: int = 10
    timeout_seconds: int = 300
    parallel_tools: bool = True
    cache_results: bool = True
    cache_ttl_hours: int = 24
    auto_fix: bool = False
    fail_on_error: bool = False
    severity_threshold: str = "info"


class QualityToolRegistry:
    """Registry of available quality tools and their configurations."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self.languages: dict[str, LanguageConfig] = {}
        self.analysis_config = QualityAnalysisConfig()
        self._initialize_default_configs()

        if config_path and config_path.exists():
            self.load_from_file(config_path)

    def _initialize_default_configs(self) -> None:
        """Initialize default tool configurations."""

        # Python tools
        python_tools = {
            "ruff": ToolConfig(
                name="ruff",
                # Lint (replaces pylint/flake8); `I` adds import sorting (replaces
                # isort). F401/F841 (unused imports/variables, ex-autoflake) are in
                # ruff's default rule set. JSON: [{code, message, filename,
                # location:{row,column}, end_location, fix, url}]. Exit 0 clean,
                # 1 findings, 2 crash/bad config.
                cmd=["ruff", "check", "--output-format", "json", "--extend-select", "I"],
                fix_cmd=["ruff", "check", "--extend-select", "I", "--fix"],
                description="Python linter and import sorter",
                supports_fix=True,
                output_format="json",
                priority=1,
            ),
            "ruff-format": ToolConfig(
                name="ruff-format",
                cmd=["ruff", "format", "--check", "--diff"],
                fix_cmd=["ruff", "format"],
                description="Python code formatter",
                supports_fix=True,
                output_format="diff",
                priority=1,
            ),
            "ty": ToolConfig(
                name="ty",
                # GitLab code-quality JSON: [{check_name, description, severity,
                # fingerprint, location:{path, positions:{begin:{line,column}}}}].
                # Exit 0 clean, 1 diagnostics, 2 crash/bad config — exit 2 can
                # still print valid JSON (e.g. a missing path), so the exit code
                # decides tool failure, not the JSON.
                cmd=["ty", "check", "--output-format", "gitlab"],
                description="Static type checker",
                supports_fix=False,
                output_format="json",
                priority=2,
                timeout=120,
            ),
            "bandit": ToolConfig(
                name="bandit",
                cmd=["bandit", "-f", "json"],
                description="Security issue scanner",
                supports_fix=False,
                output_format="json",
                priority=3,
            ),
        }

        # JavaScript/TypeScript tools
        js_ts_tools = {
            "eslint": ToolConfig(
                name="eslint",
                cmd=["eslint", "--format=json"],
                fix_cmd=["eslint", "--fix"],
                description="JavaScript/TypeScript linter",
                supports_fix=True,
                output_format="json",
                priority=1,
            ),
            "prettier": ToolConfig(
                name="prettier",
                cmd=["prettier", "--check"],
                fix_cmd=["prettier", "--write"],
                description="Code formatter",
                supports_fix=True,
                output_format="text",
                priority=1,
            ),
            "tsc": ToolConfig(
                name="tsc",
                cmd=["tsc", "--noEmit", "--pretty", "false"],
                description="TypeScript compiler",
                supports_fix=False,
                output_format="text",
                priority=2,
            ),
            "jshint": ToolConfig(
                name="jshint",
                cmd=["jshint", "--reporter=json"],
                description="JavaScript quality tool",
                supports_fix=False,
                output_format="json",
                priority=3,
            ),
        }

        # Register languages
        self.languages["python"] = LanguageConfig(
            name="python",
            extensions=[".py", ".pyx", ".pyi"],
            tools=python_tools,
            default_tools=["ruff", "ruff-format", "ty"],
            container_image="code-graph-test-python:latest",
            install_commands=["uv pip install --system ruff==0.17.3 ty==0.0.92 bandit"],
        )

        self.languages["javascript"] = LanguageConfig(
            name="javascript",
            extensions=[".js", ".jsx", ".mjs"],
            tools=js_ts_tools,
            default_tools=["eslint", "prettier"],
            container_image="code-graph-test-node:latest",
            install_commands=["npm install -g eslint prettier jshint"],
        )

        self.languages["typescript"] = LanguageConfig(
            name="typescript",
            extensions=[".ts", ".tsx"],
            tools=js_ts_tools,
            default_tools=["eslint", "prettier", "tsc"],
            container_image="code-graph-test-node:latest",
            install_commands=["npm install -g eslint prettier typescript jshint"],
        )

    def get_language_config(self, language: str) -> LanguageConfig | None:
        """Get configuration for a specific language."""
        return self.languages.get(language.lower())

    def get_tool_config(self, tool: str, language: str) -> ToolConfig | None:
        """Get configuration for a specific tool in a language."""
        lang_config = self.get_language_config(language)
        if lang_config:
            return lang_config.tools.get(tool)
        return None

    def get_tools_for_language(self, language: str) -> list[str]:
        """Get list of available tools for a language."""
        lang_config = self.get_language_config(language)
        if lang_config:
            return [name for name, tool in lang_config.tools.items() if tool.enabled]
        return []

    def get_default_tools(self, language: str) -> list[str]:
        """Get default tools for a language."""
        lang_config = self.get_language_config(language)
        if lang_config:
            return [
                tool
                for tool in lang_config.default_tools
                if tool in lang_config.tools and lang_config.tools[tool].enabled
            ]
        return []

    def supports_fix(self, tool: str, language: str) -> bool:
        """Check if a tool supports automatic fixes."""
        tool_config = self.get_tool_config(tool, language)
        return tool_config.supports_fix if tool_config else False

    def add_language(self, language_config: LanguageConfig) -> None:
        """Add a new language configuration."""
        self.languages[language_config.name.lower()] = language_config

    def add_tool(self, language: str, tool_config: ToolConfig) -> None:
        """Add a tool to a language configuration."""
        lang_config = self.get_language_config(language)
        if lang_config:
            lang_config.tools[tool_config.name] = tool_config
        else:
            logger.warning(f"Language {language} not found")

    def update_tool_config(self, language: str, tool: str, **kwargs) -> bool:
        """Update tool configuration."""
        tool_config = self.get_tool_config(tool, language)
        if tool_config:
            for key, value in kwargs.items():
                if hasattr(tool_config, key):
                    setattr(tool_config, key, value)
            return True
        return False

    def enable_tool(self, language: str, tool: str) -> bool:
        """Enable a tool."""
        return self.update_tool_config(language, tool, enabled=True)

    def disable_tool(self, language: str, tool: str) -> bool:
        """Disable a tool."""
        return self.update_tool_config(language, tool, enabled=False)

    def detect_language(self, file_path: str | Path) -> str:
        """Detect programming language from file extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        for language, config in self.languages.items():
            if extension in config.extensions:
                return language

        return "python"  # Default fallback

    def get_container_image(self, language: str) -> str:
        """Get container image for a language."""
        lang_config = self.get_language_config(language)
        return lang_config.container_image if lang_config else "code-graph-test:latest"

    def get_install_commands(self, language: str) -> list[str]:
        """Get tool installation commands for a language."""
        lang_config = self.get_language_config(language)
        return lang_config.install_commands if lang_config else []

    def load_from_file(self, config_path: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                data = json.load(f)

            # Load analysis config
            if "analysis_config" in data:
                analysis_data = data["analysis_config"]
                self.analysis_config = QualityAnalysisConfig(**analysis_data)

            # Load language configs
            if "languages" in data:
                for lang_name, lang_data in data["languages"].items():
                    # Convert tools dict
                    tools = {}
                    if "tools" in lang_data:
                        for tool_name, tool_data in lang_data["tools"].items():
                            tools[tool_name] = ToolConfig(**tool_data)

                    lang_config = LanguageConfig(
                        name=lang_data["name"],
                        extensions=lang_data["extensions"],
                        tools=tools,
                        default_tools=lang_data.get("default_tools", []),
                        container_image=lang_data.get("container_image", "code-graph-test:latest"),
                        install_commands=lang_data.get("install_commands", []),
                    )

                    self.languages[lang_name] = lang_config

            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        try:
            data = {
                "analysis_config": {
                    "max_file_size_mb": self.analysis_config.max_file_size_mb,
                    "timeout_seconds": self.analysis_config.timeout_seconds,
                    "parallel_tools": self.analysis_config.parallel_tools,
                    "cache_results": self.analysis_config.cache_results,
                    "cache_ttl_hours": self.analysis_config.cache_ttl_hours,
                    "auto_fix": self.analysis_config.auto_fix,
                    "fail_on_error": self.analysis_config.fail_on_error,
                    "severity_threshold": self.analysis_config.severity_threshold,
                },
                "languages": {},
            }

            for lang_name, lang_config in self.languages.items():
                tools_data = {}
                for tool_name, tool_config in lang_config.tools.items():
                    tools_data[tool_name] = {
                        "name": tool_config.name,
                        "cmd": tool_config.cmd,
                        "fix_cmd": tool_config.fix_cmd,
                        "description": tool_config.description,
                        "supports_fix": tool_config.supports_fix,
                        "output_format": tool_config.output_format,
                        "timeout": tool_config.timeout,
                        "enabled": tool_config.enabled,
                        "priority": tool_config.priority,
                        "config_file": tool_config.config_file,
                        "extra_args": tool_config.extra_args,
                    }

                data["languages"][lang_name] = {
                    "name": lang_config.name,
                    "extensions": lang_config.extensions,
                    "tools": tools_data,
                    "default_tools": lang_config.default_tools,
                    "container_image": lang_config.container_image,
                    "install_commands": lang_config.install_commands,
                }

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")

    def get_tool_priorities(self, language: str) -> dict[str, int]:
        """Get tool execution priorities for a language."""
        lang_config = self.get_language_config(language)
        if lang_config:
            return {name: tool.priority for name, tool in lang_config.tools.items() if tool.enabled}
        return {}

    def get_tools_by_priority(self, language: str) -> list[str]:
        """Get tools sorted by execution priority."""
        priorities = self.get_tool_priorities(language)
        return sorted(priorities.keys(), key=lambda x: priorities[x])

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate analysis config
        if self.analysis_config.max_file_size_mb <= 0:
            issues.append("max_file_size_mb must be positive")

        if self.analysis_config.timeout_seconds <= 0:
            issues.append("timeout_seconds must be positive")

        if self.analysis_config.cache_ttl_hours <= 0:
            issues.append("cache_ttl_hours must be positive")

        # Validate languages
        for lang_name, lang_config in self.languages.items():
            if not lang_config.extensions:
                issues.append(f"Language {lang_name} has no extensions defined")

            if not lang_config.tools:
                issues.append(f"Language {lang_name} has no tools defined")

            # Validate tools
            for tool_name, tool_config in lang_config.tools.items():
                if not tool_config.cmd:
                    issues.append(f"Tool {tool_name} in {lang_name} has no command defined")

                if tool_config.supports_fix and not tool_config.fix_cmd:
                    issues.append(
                        f"Tool {tool_name} in {lang_name} claims to support fixes but has no fix command"
                    )

                if tool_config.timeout <= 0:
                    issues.append(f"Tool {tool_name} in {lang_name} has invalid timeout")

        return issues

    def get_summary(self) -> dict[str, Any]:
        """Get configuration summary."""
        return {
            "total_languages": len(self.languages),
            "languages": list(self.languages.keys()),
            "total_tools": sum(len(config.tools) for config in self.languages.values()),
            "tools_by_language": {
                name: len(config.tools) for name, config in self.languages.items()
            },
            "analysis_config": {
                "max_file_size_mb": self.analysis_config.max_file_size_mb,
                "timeout_seconds": self.analysis_config.timeout_seconds,
                "parallel_tools": self.analysis_config.parallel_tools,
                "cache_results": self.analysis_config.cache_results,
            },
        }


class QualityConfigManager:
    """Manages quality configuration with persistence and validation."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.cwd() / "config"
        self.config_dir.mkdir(exist_ok=True)

        self.registry = QualityToolRegistry()
        self.config_file = self.config_dir / "quality_config.json"

        # Load existing config if available
        if self.config_file.exists():
            self.load_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        self.registry.load_from_file(self.config_file)

    def save_config(self) -> None:
        """Save configuration to file."""
        self.registry.save_to_file(self.config_file)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.registry = QualityToolRegistry()
        self.save_config()

    def backup_config(self, backup_path: Path | None = None) -> Path:
        """Create a backup of current configuration."""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_dir / f"quality_config_backup_{timestamp}.json"

        self.registry.save_to_file(backup_path)
        return backup_path

    def restore_config(self, backup_path: Path) -> None:
        """Restore configuration from backup."""
        if backup_path.exists():
            self.registry.load_from_file(backup_path)
            self.save_config()
        else:
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

    def validate_and_fix(self) -> dict[str, Any]:
        """Validate configuration and attempt to fix issues."""
        issues = self.registry.validate_config()
        fixed_issues = []

        # Attempt to fix common issues
        for issue in issues:
            if "timeout" in issue and "invalid" in issue:
                # Fix invalid timeouts
                for lang_config in self.registry.languages.values():
                    for tool_config in lang_config.tools.values():
                        if tool_config.timeout <= 0:
                            tool_config.timeout = 60
                            fixed_issues.append(f"Fixed timeout for {tool_config.name}")

        remaining_issues = self.registry.validate_config()

        return {
            "original_issues": issues,
            "fixed_issues": fixed_issues,
            "remaining_issues": remaining_issues,
            "is_valid": len(remaining_issues) == 0,
        }


# Factory functions for easy access
def get_quality_tool_registry(
    config_path: Path | None = None,
) -> QualityToolRegistry:
    """Get a quality tool registry instance."""
    return QualityToolRegistry(config_path)


def get_quality_config_manager(
    config_dir: Path | None = None,
) -> QualityConfigManager:
    """Get a quality config manager instance."""
    return QualityConfigManager(config_dir)

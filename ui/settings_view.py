"""
Settings panel — Week 4.

Lets the user edit the most important config.json values from the GUI:
- LLM provider (ollama, anthropic, gemini, grok, groq)
- Ollama model name and host
- Anthropic API key
- Authorized directories (add / remove)
- Agent parameters (max steps, planning toggle)

Changes are written back to config.json immediately. The dashboard is told
to re-build its provider when settings save (since the LLM may have changed).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QMessageBox, QComboBox, QFileDialog,
    QSpinBox, QCheckBox, QFormLayout, QGroupBox,
)
from core.cloud_llm import DEFAULT_CLOUD_MODELS


class SettingsPanel(QWidget):
    """The full Settings view."""

    settings_saved = pyqtSignal()    # main window listens to rebuild the provider

    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_ui()
        self._populate_from_config()

    def _load_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    # ---------------------- layout ----------------------

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(24, 0, 24, 16)
        main.setSpacing(12)

        # No title label — the window header already names the page.
        sub = QLabel(
            "Changes are written to config.json. Provider and model changes "
            "take effect on the next task."
        )
        sub.setProperty("role", "muted")
        sub.setWordWrap(True)
        main.addWidget(sub)

        # ───── LLM Provider section ─────
        llm_group = self._group_box("LLM Provider")
        llm_form = QFormLayout()
        self.llm_form = llm_form

        self.provider_combo = QComboBox()
        for provider in ("ollama", "anthropic", "gemini", "grok", "groq"):
            self.provider_combo.addItem(
                f"{provider} ({'local' if provider == 'ollama' else 'cloud'})", provider,
            )
        self.provider_combo.setStyleSheet(self._combo_style())
        llm_form.addRow(self._lbl("Provider"), self.provider_combo)

        # Ollama subsection
        self.ollama_host = QLineEdit()
        self.ollama_host.setStyleSheet(self._input_style())
        llm_form.addRow(self._lbl("Ollama host"), self.ollama_host)

        self.ollama_model = QLineEdit()
        self.ollama_model.setStyleSheet(self._input_style())
        llm_form.addRow(self._lbl("Ollama model"), self.ollama_model)

        self.use_vision = QCheckBox("Enable vision model (slower, requires more RAM)")
        self.use_vision.setStyleSheet("color: #ddd;")
        llm_form.addRow(self._lbl(""), self.use_vision)

        # Anthropic subsection
        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_key.setStyleSheet(self._input_style())
        self.anthropic_key.setPlaceholderText("sk-ant-...")
        llm_form.addRow(self._lbl("Anthropic API key"), self.anthropic_key)

        self.anthropic_model = QLineEdit()
        self.anthropic_model.setStyleSheet(self._input_style())
        llm_form.addRow(self._lbl("Anthropic model"), self.anthropic_model)

        self.cloud_models = {}
        for provider, default_model in DEFAULT_CLOUD_MODELS.items():
            edit = QLineEdit()
            edit.setStyleSheet(self._input_style())
            edit.setPlaceholderText(default_model)
            self.cloud_models[provider] = edit
            llm_form.addRow(self._lbl(f"{provider.title()} model"), edit)

        self.cloud_key_note = QLabel(
            "Gemini: GEMINI_API_KEY; Grok: XAI_API_KEY; Groq: GROQ_API_KEY. "
            "Launch PunkOS from the PowerShell session where you set them. "
            "These keys are not saved in config.json."
        )
        self.cloud_key_note.setWordWrap(True)
        llm_form.addRow(self.cloud_key_note)
        self.provider_combo.currentIndexChanged.connect(self._update_provider_fields)

        llm_group.layout().addLayout(llm_form)
        main.addWidget(llm_group)

        # ───── Authorized directories ─────
        dirs_group = self._group_box("Authorized directories")
        dirs_v = QVBoxLayout()

        info = QLabel("PunkOS may only read/write/delete files inside these directories.")
        info.setStyleSheet("color: #aaa; padding-bottom: 4px;")
        dirs_v.addWidget(info)

        self.dirs_list = QListWidget()
        self.dirs_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a35;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                color: #ddd;
            }
            QListWidget::item { padding: 6px; }
        """)
        self.dirs_list.setMaximumHeight(140)
        dirs_v.addWidget(self.dirs_list)

        dirs_btn_row = QHBoxLayout()
        add_btn = QPushButton("➕  Add directory")
        add_btn.setStyleSheet(self._btn_style("#4a7cd1"))
        add_btn.clicked.connect(self._on_add_dir)
        dirs_btn_row.addWidget(add_btn)

        remove_btn = QPushButton("🗑  Remove selected")
        remove_btn.setStyleSheet(self._btn_style("#a04a4a"))
        remove_btn.clicked.connect(self._on_remove_dir)
        dirs_btn_row.addWidget(remove_btn)

        dirs_btn_row.addStretch()
        dirs_v.addLayout(dirs_btn_row)
        dirs_group.layout().addLayout(dirs_v)
        main.addWidget(dirs_group)

        # ───── Agent params ─────
        agent_group = self._group_box("Agent")
        agent_form = QFormLayout()

        self.max_steps = QSpinBox()
        self.max_steps.setRange(3, 200)
        self.max_steps.setStyleSheet(self._input_style())
        agent_form.addRow(self._lbl("Max steps per task"), self.max_steps)

        self.use_planning = QCheckBox("Use upfront planning phase (recommended)")
        self.use_planning.setStyleSheet("color: #ddd;")
        agent_form.addRow(self._lbl(""), self.use_planning)

        agent_group.layout().addLayout(agent_form)
        main.addWidget(agent_group)

        main.addStretch()

        # Save / Reload row
        save_row = QHBoxLayout()
        save_row.addStretch()

        reload_btn = QPushButton("↩  Reload")
        reload_btn.setStyleSheet(self._btn_style("#666"))
        reload_btn.clicked.connect(self._on_reload)
        save_row.addWidget(reload_btn)

        save_btn = QPushButton("💾  Save settings")
        save_btn.setStyleSheet(self._btn_style("#4a7cd1"))
        save_btn.clicked.connect(self._on_save)
        save_row.addWidget(save_btn)
        main.addLayout(save_row)

    # ---------------------- styling helpers ----------------------

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; min-width: 140px;")
        return lbl

    @staticmethod
    def _input_style() -> str:
        return """
            QLineEdit, QSpinBox {
                padding: 6px 8px;
                background-color: #2a2a35;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                color: #ddd;
                min-width: 240px;
            }
        """

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                padding: 6px 8px;
                background-color: #2a2a35;
                border: 1px solid #3a3a45;
                border-radius: 4px;
                color: #ddd;
                min-width: 240px;
            }
        """

    @staticmethod
    def _btn_style(bg: str) -> str:
        return f"""
            QPushButton {{
                padding: 8px 14px;
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {bg}; opacity: 0.85; }}
        """

    @staticmethod
    def _group_box(title: str) -> QGroupBox:
        gb = QGroupBox(title)
        gb.setStyleSheet("""
            QGroupBox {
                color: #6fb1ff;
                border: 1px solid #3a3a45;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        gb.setLayout(QVBoxLayout())
        return gb

    # ---------------------- populate / collect ----------------------

    def _populate_from_config(self):
        provider = self.config.get("llm_provider", "ollama")
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(provider)))

        oc = self.config.get("ollama", {})
        self.ollama_host.setText(oc.get("host", "http://localhost:11434"))
        self.ollama_model.setText(oc.get("model", "qwen2.5:3b"))
        self.use_vision.setChecked(oc.get("use_vision", False))

        ac = self.config.get("anthropic", {})
        # Don't show placeholder API key
        api_key = ac.get("api_key", "")
        if api_key.startswith("PUT_"):
            api_key = ""
        self.anthropic_key.setText(api_key)
        self.anthropic_model.setText(ac.get("model", "claude-sonnet-4-5"))
        for provider, edit in self.cloud_models.items():
            edit.setText(self.config.get(provider, {}).get("model", DEFAULT_CLOUD_MODELS[provider]))

        self.dirs_list.clear()
        for d in self.config.get("safety", {}).get("authorized_directories", []):
            self.dirs_list.addItem(d)

        agent = self.config.get("agent", {})
        self.max_steps.setValue(agent.get("max_steps_per_task", 12))
        self.use_planning.setChecked(agent.get("use_planning_phase", True))
        self._update_provider_fields()

    def _update_provider_fields(self):
        provider = self.provider_combo.currentData()
        for field in (self.ollama_host, self.ollama_model, self.use_vision):
            self.llm_form.setRowVisible(field, provider == "ollama")
        for field in (self.anthropic_key, self.anthropic_model):
            self.llm_form.setRowVisible(field, provider == "anthropic")
        for name, field in self.cloud_models.items():
            self.llm_form.setRowVisible(field, provider == name)
        self.llm_form.setRowVisible(self.cloud_key_note, provider in self.cloud_models)

    def _collect_to_config(self) -> dict:
        cfg = copy.deepcopy(self.config)
        cfg["llm_provider"] = self.provider_combo.currentData()

        cfg.setdefault("ollama", {})
        cfg["ollama"]["host"] = self.ollama_host.text().strip()
        cfg["ollama"]["model"] = self.ollama_model.text().strip()
        cfg["ollama"]["use_vision"] = self.use_vision.isChecked()

        cfg.setdefault("anthropic", {})
        if self.anthropic_key.text().strip():
            cfg["anthropic"]["api_key"] = self.anthropic_key.text().strip()
        cfg["anthropic"]["model"] = self.anthropic_model.text().strip()
        for provider, edit in self.cloud_models.items():
            cfg.setdefault(provider, {})
            cfg[provider]["model"] = edit.text().strip() or DEFAULT_CLOUD_MODELS[provider]

        cfg.setdefault("safety", {})
        cfg["safety"]["authorized_directories"] = [
            self.dirs_list.item(i).text() for i in range(self.dirs_list.count())
        ]

        cfg.setdefault("agent", {})
        cfg["agent"]["max_steps_per_task"] = self.max_steps.value()
        cfg["agent"]["use_planning_phase"] = self.use_planning.isChecked()

        return cfg

    # ---------------------- handlers ----------------------

    def _on_add_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Add authorized directory")
        if directory:
            # Avoid duplicates
            existing = {self.dirs_list.item(i).text() for i in range(self.dirs_list.count())}
            if directory not in existing:
                self.dirs_list.addItem(directory)

    def _on_remove_dir(self):
        for item in self.dirs_list.selectedItems():
            self.dirs_list.takeItem(self.dirs_list.row(item))

    def _on_reload(self):
        self.config = self._load_config()
        self._populate_from_config()

    def _on_save(self):
        new_cfg = self._collect_to_config()
        try:
            from core.atomic_storage import write_json
            write_json(self.config_path, new_cfg)
            self.config = new_cfg
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write config.json:\n{e}")
            return

        QMessageBox.information(
            self, "Saved",
            "Settings saved to config.json.\n\n"
            "LLM changes take effect on the next task you run."
        )
        self.settings_saved.emit()

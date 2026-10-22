"""
Stimulus response reporting UI for FIF + events_log workflows.
"""

import os
import re
import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from usercontrols.pandasmodel import PandasModel
from core.instruction_embeddings import find_instruction_csv, read_instruction_csv


class _EmbeddingCreationDialog(QtWidgets.QDialog):
    """Collect provider-specific settings for a stimulus embedding export."""

    def __init__(self, names, grouping_label, default_path, parent=None, trial_df=None):
        super().__init__(parent)
        self.names = [str(value) for value in names]
        self.trial_df = (
            trial_df.copy()
            if trial_df is not None and not getattr(trial_df, "empty", True)
            else pd.DataFrame()
        )
        self._automatic_output_base_path = Path(str(default_path))
        self._output_path_is_automatic = True
        self.setWindowTitle("Create stimulus embeddings")
        self.setModal(True)
        self.resize(720, 440)
        self.setMinimumWidth(660)

        root = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Create one semantic vector for each label in the current Stimulus Summary. "
            "The exported JSON can be selected in the Concept decoding tab."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        common_form = QtWidgets.QFormLayout()
        self.lbl_source = QtWidgets.QLabel(
            f"Current summary — {grouping_label} ({len(self.names)} unique labels)"
        )
        self.lbl_source.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        common_form.addRow("Labels", self.lbl_source)

        self.cmb_provider = QtWidgets.QComboBox()
        self.cmb_provider.addItem("OpenAI text embedding (default)", "openai")
        self.cmb_provider.addItem("Word2Vec / KeyedVectors model", "word2vec")
        self.cmb_provider.addItem("CLIP image embedding", "clip_image")
        self.cmb_provider.addItem("CLIP text embedding", "clip_text")
        self.cmb_provider.addItem("DINOv2 image embedding", "dinov2_image")
        common_form.addRow("Embedding generator", self.cmb_provider)
        root.addLayout(common_form)

        self.provider_stack = QtWidgets.QStackedWidget()

        openai_page = QtWidgets.QWidget()
        openai_form = QtWidgets.QFormLayout(openai_page)
        self.cmb_openai_model = QtWidgets.QComboBox()
        self.cmb_openai_model.setEditable(True)
        self.cmb_openai_model.addItems(["text-embedding-3-large", "text-embedding-3-small"])
        self.cmb_openai_model.setToolTip("The model name is editable for compatible embedding endpoints.")
        openai_form.addRow("Model", self.cmb_openai_model)
        self.spb_openai_dimensions = QtWidgets.QSpinBox()
        self.spb_openai_dimensions.setRange(0, 3072)
        self.spb_openai_dimensions.setSpecialValueText("Native")
        self.spb_openai_dimensions.setValue(0)
        self.spb_openai_dimensions.setToolTip(
            "Use Native unless a smaller supported output dimension is specifically needed."
        )
        openai_form.addRow("Dimensions", self.spb_openai_dimensions)
        openai_note = QtWidgets.QLabel(
            "Uses OPENAI_API_KEY, or asks for a key after Create is pressed."
        )
        openai_note.setWordWrap(True)
        openai_form.addRow("", openai_note)
        self.provider_stack.addWidget(openai_page)

        word2vec_page = QtWidgets.QWidget()
        word2vec_form = QtWidgets.QFormLayout(word2vec_page)
        word2vec_path_widget = QtWidgets.QWidget()
        word2vec_path_layout = QtWidgets.QHBoxLayout(word2vec_path_widget)
        word2vec_path_layout.setContentsMargins(0, 0, 0, 0)
        self.le_word2vec_path = QtWidgets.QLineEdit()
        self.le_word2vec_path.setPlaceholderText(
            "Pretrained .bin, .vec, .txt, .kv, or .model file"
        )
        word2vec_path_layout.addWidget(self.le_word2vec_path)
        self.btn_word2vec_browse = QtWidgets.QPushButton("Browse")
        self.btn_word2vec_browse.clicked.connect(self._browse_word2vec_model)
        word2vec_path_layout.addWidget(self.btn_word2vec_browse)
        word2vec_form.addRow("Model file", word2vec_path_widget)

        self.cmb_word2vec_aggregation = QtWidgets.QComboBox()
        self.cmb_word2vec_aggregation.addItem(
            "Exact phrase, otherwise mean available words", "exact_or_mean"
        )
        self.cmb_word2vec_aggregation.addItem("Mean available word vectors", "mean")
        word2vec_form.addRow("Multiword labels", self.cmb_word2vec_aggregation)

        self.cmb_word2vec_oov = QtWidgets.QComboBox()
        self.cmb_word2vec_oov.addItem("Stop and report unavailable labels", "error")
        self.cmb_word2vec_oov.addItem("Skip labels with no usable words", "skip")
        word2vec_form.addRow("Unavailable labels", self.cmb_word2vec_oov)
        word2vec_note = QtWidgets.QLabel(
            "Uses a pretrained model rather than training on the short stimulus list. "
            "Coverage and partially matched labels are recorded in the output metadata."
        )
        word2vec_note.setWordWrap(True)
        word2vec_form.addRow("", word2vec_note)
        self.provider_stack.addWidget(word2vec_page)

        clip_image_page = QtWidgets.QWidget()
        clip_image_form = QtWidgets.QFormLayout(clip_image_page)
        self.cmb_clip_image_model = QtWidgets.QComboBox()
        self.cmb_clip_image_model.setEditable(True)
        self.cmb_clip_image_model.addItems(
            ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
        )
        clip_image_form.addRow("Model", self.cmb_clip_image_model)
        clip_image_root_widget = QtWidgets.QWidget()
        clip_image_root_layout = QtWidgets.QHBoxLayout(clip_image_root_widget)
        clip_image_root_layout.setContentsMargins(0, 0, 0, 0)
        self.le_clip_image_root = QtWidgets.QLineEdit()
        self.le_clip_image_root.setPlaceholderText("Folder containing the presented stimulus images")
        clip_image_root_layout.addWidget(self.le_clip_image_root)
        self.btn_clip_image_root = QtWidgets.QPushButton("Browse")
        self.btn_clip_image_root.clicked.connect(
            lambda: self._browse_image_root(self.le_clip_image_root)
        )
        clip_image_root_layout.addWidget(self.btn_clip_image_root)
        clip_image_form.addRow("Stimulus folder", clip_image_root_widget)
        self.cmb_clip_image_missing = QtWidgets.QComboBox()
        self.cmb_clip_image_missing.addItem("Stop and report missing/ambiguous images", "error")
        self.cmb_clip_image_missing.addItem("Skip labels without an unambiguous image", "skip")
        clip_image_form.addRow("Unavailable labels", self.cmb_clip_image_missing)
        self.cmb_clip_image_device = self._device_combo()
        clip_image_form.addRow("Device", self.cmb_clip_image_device)
        clip_image_note = QtWidgets.QLabel(
            "Embeds each actual image, L2-normalizes it, then averages variants into "
            "the currently selected summary label. Model weights download on first use."
        )
        clip_image_note.setWordWrap(True)
        clip_image_form.addRow("", clip_image_note)
        self.provider_stack.addWidget(clip_image_page)

        clip_text_page = QtWidgets.QWidget()
        clip_text_form = QtWidgets.QFormLayout(clip_text_page)
        self.cmb_clip_text_model = QtWidgets.QComboBox()
        self.cmb_clip_text_model.setEditable(True)
        self.cmb_clip_text_model.addItems(
            ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
        )
        clip_text_form.addRow("Model", self.cmb_clip_text_model)
        self.le_clip_text_prompt = QtWidgets.QLineEdit("a photo of a {label}")
        self.le_clip_text_prompt.setToolTip("The prompt must include the {label} placeholder.")
        clip_text_form.addRow("Prompt", self.le_clip_text_prompt)
        self.cmb_clip_text_device = self._device_combo()
        clip_text_form.addRow("Device", self.cmb_clip_text_device)
        clip_text_note = QtWidgets.QLabel(
            "Creates label embeddings in the same joint space as CLIP images. "
            "Model weights download on first use."
        )
        clip_text_note.setWordWrap(True)
        clip_text_form.addRow("", clip_text_note)
        self.provider_stack.addWidget(clip_text_page)

        dinov2_page = QtWidgets.QWidget()
        dinov2_form = QtWidgets.QFormLayout(dinov2_page)
        self.cmb_dinov2_model = QtWidgets.QComboBox()
        self.cmb_dinov2_model.setEditable(True)
        self.cmb_dinov2_model.addItems(
            ["facebook/dinov2-base", "facebook/dinov2-small", "facebook/dinov2-large"]
        )
        dinov2_form.addRow("Model", self.cmb_dinov2_model)
        dinov2_root_widget = QtWidgets.QWidget()
        dinov2_root_layout = QtWidgets.QHBoxLayout(dinov2_root_widget)
        dinov2_root_layout.setContentsMargins(0, 0, 0, 0)
        self.le_dinov2_image_root = QtWidgets.QLineEdit()
        self.le_dinov2_image_root.setPlaceholderText("Folder containing the presented stimulus images")
        dinov2_root_layout.addWidget(self.le_dinov2_image_root)
        self.btn_dinov2_image_root = QtWidgets.QPushButton("Browse")
        self.btn_dinov2_image_root.clicked.connect(
            lambda: self._browse_image_root(self.le_dinov2_image_root)
        )
        dinov2_root_layout.addWidget(self.btn_dinov2_image_root)
        dinov2_form.addRow("Stimulus folder", dinov2_root_widget)
        self.cmb_dinov2_missing = QtWidgets.QComboBox()
        self.cmb_dinov2_missing.addItem("Stop and report missing/ambiguous images", "error")
        self.cmb_dinov2_missing.addItem("Skip labels without an unambiguous image", "skip")
        dinov2_form.addRow("Unavailable labels", self.cmb_dinov2_missing)
        self.cmb_dinov2_device = self._device_combo()
        dinov2_form.addRow("Device", self.cmb_dinov2_device)
        dinov2_note = QtWidgets.QLabel(
            "Creates visual-only DINOv2 prototypes from the actual images. "
            "Model weights download on first use."
        )
        dinov2_note.setWordWrap(True)
        dinov2_form.addRow("", dinov2_note)
        self.provider_stack.addWidget(dinov2_page)
        root.addWidget(self.provider_stack)

        output_widget = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.le_output_path = QtWidgets.QLineEdit()
        self.le_output_path.textEdited.connect(self._mark_output_path_as_custom)
        output_layout.addWidget(self.le_output_path)
        self.btn_output_browse = QtWidgets.QPushButton("Browse")
        self.btn_output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(self.btn_output_browse)
        output_form = QtWidgets.QFormLayout()
        output_form.addRow("Output JSON", output_widget)
        self.le_instruction_csv = QtWidgets.QLineEdit(
            find_instruction_csv(Path(default_path).parent)
        )
        self.le_instruction_csv.setPlaceholderText("Optional block-instructions.csv")
        instruction_widget = QtWidgets.QWidget()
        instruction_layout = QtWidgets.QHBoxLayout(instruction_widget)
        instruction_layout.setContentsMargins(0, 0, 0, 0)
        instruction_layout.addWidget(self.le_instruction_csv)
        instruction_browse = QtWidgets.QPushButton("Browse")
        instruction_browse.clicked.connect(self._browse_instruction_csv)
        instruction_layout.addWidget(instruction_browse)
        output_form.addRow("Block instructions", instruction_widget)
        instruction_note = QtWidgets.QLabel(
            "Includes unique line_2 text as test-time instruction targets. "
            "DINOv2 cannot embed text; its exports contain image targets only."
        )
        instruction_note.setWordWrap(True)
        output_form.addRow("", instruction_note)

        # Residualization section
        self.chk_residualize = QtWidgets.QCheckBox(
            "Residualize embeddings against categories/tasks"
        )
        self.chk_residualize.setToolTip(
            "Regress out superordinate category or task-decision confounds "
            "so the resulting concept embeddings represent concept-unique features."
        )
        output_form.addRow("Residualization", self.chk_residualize)

        self.residualize_container = QtWidgets.QWidget()
        res_layout = QtWidgets.QFormLayout(self.residualize_container)
        res_layout.setContentsMargins(0, 0, 0, 0)

        self.cmb_residualize_source = QtWidgets.QComboBox()
        self.cmb_residualize_source.addItem(
            "Stimulus category (from trial table stimulus_group)", "category"
        )
        self.cmb_residualize_source.addItem(
            "Block instruction task (from block-instructions.csv)", "instruction"
        )
        self.cmb_residualize_source.addItem(
            "Custom category mapping CSV / TSV file", "custom_csv"
        )
        self.cmb_residualize_source.setToolTip(
            "Select whether to residualize against the stimulus category "
            "or the block-level semantic decision task."
        )
        res_layout.addRow("Confound source", self.cmb_residualize_source)

        cat_csv_widget = QtWidgets.QWidget()
        cat_csv_layout = QtWidgets.QHBoxLayout(cat_csv_widget)
        cat_csv_layout.setContentsMargins(0, 0, 0, 0)
        self.le_category_csv = QtWidgets.QLineEdit()
        self.le_category_csv.setPlaceholderText("Optional concept-to-category CSV (concept,category)")
        cat_csv_layout.addWidget(self.le_category_csv)
        self.btn_category_csv_browse = QtWidgets.QPushButton("Browse")
        self.btn_category_csv_browse.clicked.connect(self._browse_category_csv)
        cat_csv_layout.addWidget(self.btn_category_csv_browse)
        res_layout.addRow("Category CSV", cat_csv_widget)

        res_note = QtWidgets.QLabel(
            "Removes the category centroid or task-decision projection, ensuring models "
            "cannot rely on 'animal vs non-animal' decisions to predict concept embeddings."
        )
        res_note.setWordWrap(True)
        res_layout.addRow("", res_note)

        output_form.addRow("", self.residualize_container)
        self.residualize_container.setVisible(False)
        self.chk_residualize.toggled.connect(self.residualize_container.setVisible)
        self.chk_residualize.toggled.connect(self._refresh_automatic_output_path)
        self.cmb_residualize_source.currentIndexChanged.connect(self._refresh_automatic_output_path)
        root.addLayout(output_form)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setText("Create")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self.cmb_provider.currentIndexChanged.connect(self.provider_stack.setCurrentIndex)
        self.cmb_provider.currentIndexChanged.connect(self._refresh_automatic_output_path)
        self.cmb_openai_model.currentTextChanged.connect(
            self._refresh_automatic_output_path
        )
        self.spb_openai_dimensions.valueChanged.connect(
            self._refresh_automatic_output_path
        )
        self.le_word2vec_path.textChanged.connect(self._refresh_automatic_output_path)
        self.cmb_clip_image_model.currentTextChanged.connect(
            self._refresh_automatic_output_path
        )
        self.cmb_clip_text_model.currentTextChanged.connect(
            self._refresh_automatic_output_path
        )
        self.cmb_dinov2_model.currentTextChanged.connect(
            self._refresh_automatic_output_path
        )
        self.provider_stack.setCurrentIndex(0)
        self._refresh_automatic_output_path()

    @staticmethod
    def _filename_component(value, fallback="model"):
        component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
        component = re.sub(r"-+", "-", component).strip("-._")
        return component or fallback

    def _selected_embedding_tag(self):
        provider = str(self.cmb_provider.currentData() or "openai")
        if provider == "word2vec":
            model_name = Path(str(self.le_word2vec_path.text() or "")).name
            while Path(model_name).suffix.lower() in {
                ".bin",
                ".gz",
                ".kv",
                ".kvs",
                ".model",
                ".txt",
                ".vec",
            }:
                model_name = Path(model_name).stem
            return "word2vec_" + self._filename_component(
                model_name, "model-not-selected"
            )
        if provider == "clip_image":
            model_name = self.cmb_clip_image_model.currentText()
            provider_tag = "clip-image"
        elif provider == "clip_text":
            model_name = self.cmb_clip_text_model.currentText()
            provider_tag = "clip-text"
        elif provider == "dinov2_image":
            model_name = self.cmb_dinov2_model.currentText()
            provider_tag = "dinov2-image"
        else:
            model_name = self.cmb_openai_model.currentText()
            provider_tag = "openai"
        model_tag = self._filename_component(model_name)
        if provider == "openai" and int(self.spb_openai_dimensions.value()) > 0:
            model_tag += f"-d{int(self.spb_openai_dimensions.value())}"
        tag = f"{provider_tag}_{model_tag}"
        if getattr(self, "chk_residualize", None) and self.chk_residualize.isChecked():
            tag += "-residualized"
        return tag

    def _refresh_automatic_output_path(self, *_args):
        if not self._output_path_is_automatic:
            return
        base_path = self._automatic_output_base_path
        prefix = "concept_embeddings_"
        if base_path.stem.startswith(prefix):
            remainder = base_path.stem[len(prefix) :]
            filename = f"{prefix}{self._selected_embedding_tag()}_{remainder}"
        else:
            filename = f"{base_path.stem}_{self._selected_embedding_tag()}"
        suffix = base_path.suffix or ".json"
        self.le_output_path.setText(str(base_path.with_name(filename + suffix)))

    def _mark_output_path_as_custom(self, *_args):
        self._output_path_is_automatic = False

    @staticmethod
    def _device_combo():
        combo = QtWidgets.QComboBox()
        combo.addItem("Auto", "auto")
        combo.addItem("CPU", "cpu")
        combo.addItem("Apple Metal (MPS)", "mps")
        combo.addItem("CUDA", "cuda")
        return combo

    def _browse_image_root(self, line_edit):
        start_path = str(line_edit.text() or os.getcwd())
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select stimulus image folder",
            str(Path(start_path).expanduser()),
        )
        if path:
            line_edit.setText(str(path))

    def _browse_word2vec_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select pretrained Word2Vec model",
            str(Path(self.le_word2vec_path.text() or os.getcwd()).expanduser()),
            "Word2Vec models (*.bin *.bin.gz *.vec *.vec.gz *.txt *.txt.gz *.gz *.kv *.kvs *.model);;All files (*.*)",
        )
        if path:
            self.le_word2vec_path.setText(str(path))

    def _browse_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save concept embeddings dictionary",
            str(Path(self.le_output_path.text() or os.getcwd()).expanduser()),
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            if not str(path).lower().endswith(".json"):
                path = f"{path}.json"
            self._output_path_is_automatic = False
            self.le_output_path.setText(str(path))

    def _browse_instruction_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select block instructions", self.le_instruction_csv.text(), "CSV files (*.csv)"
        )
        if path:
            self.le_instruction_csv.setText(path)

    def _browse_category_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select category mapping file",
            str(Path(self.le_category_csv.text() or os.getcwd()).expanduser()),
            "Table files (*.csv *.tsv *.txt);;All files (*.*)",
        )
        if path:
            self.le_category_csv.setText(str(path))

    def accept(self):
        instruction_path = self.le_instruction_csv.text().strip()
        if instruction_path:
            try:
                read_instruction_csv(os.path.expanduser(instruction_path))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid block instructions", str(exc))
                return
        output_path = str(self.le_output_path.text() or "").strip()
        if not output_path:
            QtWidgets.QMessageBox.warning(self, "Output required", "Choose an output JSON file.")
            return
        provider = str(self.cmb_provider.currentData() or "openai")
        if provider == "openai":
            if not str(self.cmb_openai_model.currentText() or "").strip():
                QtWidgets.QMessageBox.warning(self, "Model required", "Enter an OpenAI embedding model.")
                return
        elif provider == "word2vec":
            model_path = os.path.abspath(
                os.path.expanduser(str(self.le_word2vec_path.text() or "").strip())
            )
            if not os.path.isfile(model_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Model file required",
                    "Choose an existing Word2Vec/KeyedVectors model file.",
                )
                return
        elif provider == "clip_text":
            if not str(self.cmb_clip_text_model.currentText() or "").strip():
                QtWidgets.QMessageBox.warning(self, "Model required", "Enter a CLIP model.")
                return
            if "{label}" not in str(self.le_clip_text_prompt.text() or ""):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Prompt placeholder required",
                    "The CLIP text prompt must contain {label}.",
                )
                return
        elif provider in {"clip_image", "dinov2_image"}:
            if provider == "clip_image":
                model = str(self.cmb_clip_image_model.currentText() or "").strip()
                image_root = str(self.le_clip_image_root.text() or "").strip()
            else:
                model = str(self.cmb_dinov2_model.currentText() or "").strip()
                image_root = str(self.le_dinov2_image_root.text() or "").strip()
            if not model:
                QtWidgets.QMessageBox.warning(self, "Model required", "Enter an image model.")
                return
            if not os.path.isdir(os.path.abspath(os.path.expanduser(image_root))):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Stimulus folder required",
                    "Choose the folder containing the presented stimulus images.",
                )
                return
        if self.chk_residualize.isChecked():
            source = str(self.cmb_residualize_source.currentData() or "category")
            cat_csv = str(self.le_category_csv.text() or "").strip()
            if source == "custom_csv":
                if not cat_csv or not os.path.isfile(os.path.abspath(os.path.expanduser(cat_csv))):
                    QtWidgets.QMessageBox.warning(
                        self, "Category CSV required", "Choose an existing category mapping CSV/TSV file."
                    )
                    return
            elif source == "instruction":
                instruction_path = str(self.le_instruction_csv.text() or "").strip()
                if not instruction_path or not os.path.isfile(os.path.abspath(os.path.expanduser(instruction_path))):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Block instructions required",
                        "Residualizing against instruction tasks requires a valid block-instructions.csv file.",
                    )
                    return
            elif source == "category":
                has_trial_cat = (
                    not getattr(self, "trial_df", pd.DataFrame()).empty
                    and any(
                        col in self.trial_df.columns
                        for col in ("stimulus_group", "category", "category_label", "group")
                    )
                )
                if not has_trial_cat and not (cat_csv and os.path.isfile(os.path.abspath(os.path.expanduser(cat_csv)))):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Category source missing",
                        "The current trial summary does not contain a stimulus_group/category column. "
                        "Please provide a Category CSV mapping each stimulus to its category, "
                        "or choose another confound source.",
                    )
                    return
        super().accept()

    def configuration(self):
        dimensions = int(self.spb_openai_dimensions.value())
        output_path = str(self.le_output_path.text() or "").strip()
        if not output_path.lower().endswith(".json"):
            output_path = f"{output_path}.json"
        provider = str(self.cmb_provider.currentData() or "openai")
        config = {
            "instruction_csv": self.le_instruction_csv.text().strip(),
            "provider": provider,
            "openai_model": str(self.cmb_openai_model.currentText() or "").strip(),
            "dimensions": dimensions if dimensions > 0 else None,
            "word2vec_model_path": os.path.abspath(
                os.path.expanduser(str(self.le_word2vec_path.text() or "").strip())
            ),
            "word2vec_aggregation": str(
                self.cmb_word2vec_aggregation.currentData() or "exact_or_mean"
            ),
            "word2vec_oov_policy": str(self.cmb_word2vec_oov.currentData() or "error"),
            "output_path": output_path,
            "residualize": bool(self.chk_residualize.isChecked()),
            "residualize_source": str(self.cmb_residualize_source.currentData() or "category"),
            "category_csv": str(self.le_category_csv.text() or "").strip(),
        }
        if provider == "clip_image":
            config.update(
                {
                    "transformer_model": str(self.cmb_clip_image_model.currentText() or "").strip(),
                    "image_root": os.path.abspath(
                        os.path.expanduser(str(self.le_clip_image_root.text() or "").strip())
                    ),
                    "image_missing_policy": str(
                        self.cmb_clip_image_missing.currentData() or "error"
                    ),
                    "device": str(self.cmb_clip_image_device.currentData() or "auto"),
                }
            )
        elif provider == "clip_text":
            config.update(
                {
                    "transformer_model": str(self.cmb_clip_text_model.currentText() or "").strip(),
                    "prompt_template": str(self.le_clip_text_prompt.text() or ""),
                    "device": str(self.cmb_clip_text_device.currentData() or "auto"),
                }
            )
        elif provider == "dinov2_image":
            config.update(
                {
                    "transformer_model": str(self.cmb_dinov2_model.currentText() or "").strip(),
                    "image_root": os.path.abspath(
                        os.path.expanduser(str(self.le_dinov2_image_root.text() or "").strip())
                    ),
                    "image_missing_policy": str(self.cmb_dinov2_missing.currentData() or "error"),
                    "device": str(self.cmb_dinov2_device.currentData() or "auto"),
                }
            )
        return config


class _ConceptEmbeddingWorker(QtCore.QObject):
    finished = QtCore.Signal(str, int, int, str)
    failed = QtCore.Signal(str)

    def __init__(self, names, output_path, config, api_key=""):
        super().__init__()
        self.names = [str(v) for v in names]
        self.output_path = str(output_path)
        self.config = dict(config or {})
        self.api_key = str(api_key or "")

    @staticmethod
    def _load_embedding_script():
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "create_concept_embedding_dictionary.py"
        spec = importlib.util.spec_from_file_location(
            "mas_create_concept_embedding_dictionary",
            str(script_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load embedding helper script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run(self):
        try:
            helper = self._load_embedding_script()
            provider = str(self.config.get("provider", "openai") or "openai").strip().lower()
            output_names = list(self.names)
            metadata = {
                "provider": provider,
                "requested_labels": int(len(self.names)),
                "source_grouping": str(self.config.get("source_grouping", "")),
            }
            details = ""
            instruction_path = str(self.config.get("instruction_csv", "")).strip()
            instructions = read_instruction_csv(os.path.expanduser(instruction_path)) if instruction_path else None
            if provider == "word2vec":
                output_names, embeddings, word2vec_metadata = helper.create_word2vec_embeddings(
                    self.names,
                    model_path=str(self.config.get("word2vec_model_path", "")),
                    aggregation=str(
                        self.config.get("word2vec_aggregation", "exact_or_mean")
                    ),
                    oov_policy=str(self.config.get("word2vec_oov_policy", "error")),
                )
                model = str(word2vec_metadata.get("model", "Word2Vec"))
                metadata.update(word2vec_metadata)
                missing = list(word2vec_metadata.get("missing_labels", []))
                partial = dict(word2vec_metadata.get("partial_labels", {}))
                detail_parts = [f"{len(output_names)}/{len(self.names)} requested labels embedded"]
                if missing:
                    detail_parts.append(f"{len(missing)} unavailable")
                if partial:
                    detail_parts.append(f"{len(partial)} partial token matches")
                details = "; ".join(detail_parts)
            elif provider == "openai":
                model = str(
                    self.config.get("openai_model", helper.DEFAULT_MODEL) or helper.DEFAULT_MODEL
                )
                embeddings = helper.create_embeddings(
                    self.names,
                    api_key=self.api_key,
                    endpoint=helper.DEFAULT_ENDPOINT,
                    model=model,
                    dimensions=self.config.get("dimensions"),
                    batch_size=64,
                    timeout=120.0,
                    max_retries=5,
                )
                metadata.update(
                    {
                        "model": model,
                        "dimensions_requested": self.config.get("dimensions"),
                        "embedded_labels": int(len(output_names)),
                    }
                )
                details = f"{len(output_names)} requested labels embedded with {model}"
            elif provider == "clip_text":
                model = str(
                    self.config.get("transformer_model", helper.DEFAULT_CLIP_MODEL)
                    or helper.DEFAULT_CLIP_MODEL
                )
                output_names, embeddings, generated_metadata = (
                    helper.create_clip_text_embeddings(
                        self.names,
                        model_name=model,
                        prompt_template=str(
                            self.config.get("prompt_template", "a photo of a {label}")
                        ),
                        device=str(self.config.get("device", "auto")),
                        batch_size=32,
                    )
                )
                metadata.update(generated_metadata)
                details = f"{len(output_names)} labels embedded with CLIP text ({model})"
            elif provider in {"clip_image", "dinov2_image"}:
                default_model = (
                    helper.DEFAULT_CLIP_MODEL
                    if provider == "clip_image"
                    else helper.DEFAULT_DINOV2_MODEL
                )
                model = str(
                    self.config.get("transformer_model", default_model) or default_model
                )
                output_names, embeddings, generated_metadata = helper.create_image_embeddings(
                    self.names,
                    image_references_by_label=dict(
                        self.config.get("image_references_by_label", {})
                    ),
                    image_root=str(self.config.get("image_root", "")),
                    provider=provider,
                    model_name=model,
                    missing_policy=str(
                        self.config.get("image_missing_policy", "error")
                    ),
                    device=str(self.config.get("device", "auto")),
                    batch_size=16,
                )
                metadata.update(generated_metadata)
                missing = list(generated_metadata.get("missing_labels", []))
                image_count = int(generated_metadata.get("resolved_image_count", 0))
                detail_parts = [
                    f"{len(output_names)}/{len(self.names)} labels embedded from {image_count} images"
                ]
                if missing:
                    detail_parts.append(f"{len(missing)} labels skipped")
                details = "; ".join(detail_parts)
            else:
                raise ValueError(f"Unsupported embedding generator: {provider!r}.")
            args = SimpleNamespace(
                provider=provider,
                model=model,
                input="stimulus_summary",
                column="stimulus",
            )
            if instructions:
                instruction_names = instructions["names"]
                if provider == "dinov2_image":
                    instructions["status"] = "unsupported_text_encoder"
                    details += "; instruction text unavailable for DINOv2"
                else:
                    if provider == "openai":
                        instruction_vectors = helper.create_embeddings(
                            instruction_names, api_key=self.api_key,
                            endpoint=helper.DEFAULT_ENDPOINT, model=model,
                            dimensions=self.config.get("dimensions"), batch_size=64,
                            timeout=120.0, max_retries=5,
                        )
                    elif provider == "word2vec":
                        instruction_names, instruction_vectors, instruction_meta = helper.create_word2vec_embeddings(
                            instruction_names, model_path=self.config["word2vec_model_path"],
                            aggregation=self.config.get("word2vec_aggregation", "exact_or_mean"),
                            oov_policy=self.config.get("word2vec_oov_policy", "error"),
                        )
                        instructions["coverage"] = instruction_meta
                    else:
                        instruction_names, instruction_vectors, _ = helper.create_clip_text_embeddings(
                            instruction_names, model_name=model, prompt_template="{label}",
                            device=self.config.get("device", "auto"),
                        )
                    instructions.update(status="embedded", names=instruction_names,
                                        embeddings=instruction_vectors, model=model,
                                        provider="clip_text" if provider.startswith("clip_") else provider)
                    details += f"; {len(instruction_names)} instruction targets embedded"
                metadata["instruction_targets"] = instructions

            if bool(self.config.get("residualize", False)):
                from core.concept_embedding_residualization import (
                    TargetEmbeddingResidualizer,
                    resolve_concept_confounds,
                )
                confound_source = str(self.config.get("residualize_source", "category"))
                cat_csv = str(self.config.get("category_csv", "")).strip()
                trial_df = self.config.get("trial_df")
                inst_p = str(self.config.get("instruction_csv", "")).strip()

                concept_to_confound = resolve_concept_confounds(
                    output_names,
                    trial_df=trial_df,
                    confound_source=confound_source,
                    category_csv=cat_csv,
                    instruction_path=inst_p,
                )
                confounds = [concept_to_confound[name] for name in output_names]
                residualizer = TargetEmbeddingResidualizer(include_intercept=True, l2_normalize_output=True)
                raw_emb_arr = np.asarray(embeddings, dtype=np.float32)
                res_embeddings = residualizer.fit_transform(raw_emb_arr, confounds)
                embeddings = res_embeddings.tolist() if isinstance(embeddings, list) else res_embeddings
                metadata["residualized"] = True
                metadata["residualization_variance_explained"] = float(residualizer.variance_explained_)
                metadata["confound_categories"] = residualizer.categories_
                metadata["residualization_source"] = confound_source
                details += (
                    f"; residualized against {confound_source} "
                    f"({residualizer.variance_explained_ * 100:.1f}% variance explained)"
                )

            helper.write_output(
                Path(self.output_path),
                output_names,
                embeddings,
                args,
                metadata=metadata,
            )
            dim = int(len(embeddings[0])) if embeddings else 0
            self.finished.emit(self.output_path, int(len(output_names)), dim, details)
        except Exception as exc:
            self.failed.emit(str(exc))


class StimulusResponseReport:
    """Display trial-level and stimulus-level responsiveness tables."""

    def __init__(self, report_view, c=None):
        self.report_view = report_view
        self.c = c
        self._trial_df_raw = pd.DataFrame()
        self._summary_df = pd.DataFrame()
        self._metadata = {}
        self._trial_df_current = pd.DataFrame()
        self.trials_model = None
        self.summary_model = None
        self._embedding_thread = None
        self._embedding_worker = None
        self._init_layout()

    def _init_layout(self):
        self.widget = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.widget.setLayout(root)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)

        controls.addWidget(QtWidgets.QLabel("Score threshold"))
        self.spb_score_thr = QtWidgets.QDoubleSpinBox()
        self.spb_score_thr.setDecimals(3)
        self.spb_score_thr.setRange(-10.0, 10.0)
        self.spb_score_thr.setSingleStep(0.05)
        self.spb_score_thr.setValue(0.15)
        controls.addWidget(self.spb_score_thr)

        controls.addWidget(QtWidgets.QLabel("Responsive rate %"))
        self.spb_rate_thr = QtWidgets.QDoubleSpinBox()
        self.spb_rate_thr.setDecimals(1)
        self.spb_rate_thr.setRange(0.0, 100.0)
        self.spb_rate_thr.setSingleStep(5.0)
        self.spb_rate_thr.setValue(35.0)
        controls.addWidget(self.spb_rate_thr)

        controls.addWidget(QtWidgets.QLabel("Min trials"))
        self.spb_min_trials = QtWidgets.QSpinBox()
        self.spb_min_trials.setRange(1, 1000)
        self.spb_min_trials.setSingleStep(1)
        self.spb_min_trials.setValue(3)
        controls.addWidget(self.spb_min_trials)

        controls.addWidget(QtWidgets.QLabel("Summary by"))
        self.cmb_grouping = QtWidgets.QComboBox()
        self.cmb_grouping.addItems(["Canonical stimulus", "Stimulus category", "Original stimulus"])
        self.cmb_grouping.currentIndexChanged.connect(self.recompute_summary)
        controls.addWidget(self.cmb_grouping)

        self.btn_recompute = QtWidgets.QPushButton("Recompute")
        self.btn_recompute.clicked.connect(self.recompute_summary)
        controls.addWidget(self.btn_recompute)

        self.btn_export = QtWidgets.QPushButton("Export Report")
        self.btn_export.clicked.connect(self.export_report)
        controls.addWidget(self.btn_export)

        self.btn_export_trial_table = QtWidgets.QPushButton("Export Trial Table")
        self.btn_export_trial_table.setToolTip(
            "Export the mapped per-trial table used by command-line concept decoding "
            "(source_fif_path, onset_sample, stimulus labels, and run metadata)."
        )
        self.btn_export_trial_table.clicked.connect(self.export_trial_table)
        controls.addWidget(self.btn_export_trial_table)

        self.btn_create_embeddings = QtWidgets.QPushButton("Create Embeddings")
        self.btn_create_embeddings.setToolTip(
            "Create OpenAI, Word2Vec, CLIP, or DINOv2 JSON embeddings for the current Stimulus Summary."
        )
        self.btn_create_embeddings.clicked.connect(self.create_embeddings_from_summary)
        controls.addWidget(self.btn_create_embeddings)

        controls.addStretch()
        root.addLayout(controls)

        self.status_label = QtWidgets.QLabel("No stimulus-response report loaded.")
        root.addWidget(self.status_label)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self.tbl_trials = QtWidgets.QTableView()
        self.tbl_trials.setSortingEnabled(True)
        self.tbl_trials.setAlternatingRowColors(True)
        self.tabs.addTab(self.tbl_trials, "Trials")

        self.tbl_summary = QtWidgets.QTableView()
        self.tbl_summary.setSortingEnabled(True)
        self.tbl_summary.setAlternatingRowColors(True)
        self.tabs.addTab(self.tbl_summary, "Stimulus Summary")

        if self.report_view is not None:
            self.report_view.addWidget(self.widget)

    def clear(self, message="No stimulus-response report loaded."):
        self._trial_df_raw = pd.DataFrame()
        self._summary_df = pd.DataFrame()
        self._metadata = {}
        self._trial_df_current = pd.DataFrame()
        self.tbl_trials.setModel(None)
        self.tbl_summary.setModel(None)
        self.status_label.setText(str(message))

    def set_trial_data(self, trial_df, metadata=None):
        if metadata is None:
            metadata = {}
        self._metadata = dict(metadata)
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            self.clear("No trial data available from events_log + FIF mapping.")
            return

        self._trial_df_raw = trial_df.copy()
        if "stimulus" in self._trial_df_raw.columns:
            if "canonical_stimulus" not in self._trial_df_raw.columns:
                self._trial_df_raw["canonical_stimulus"] = self._trial_df_raw["stimulus"].map(
                    self._canonicalize_stimulus_label
                )
            else:
                self._trial_df_raw["canonical_stimulus"] = self._trial_df_raw["canonical_stimulus"].fillna(
                    self._trial_df_raw["stimulus"].map(self._canonicalize_stimulus_label)
                )
        self.recompute_summary()

    @staticmethod
    def _canonicalize_stimulus_label(stimulus_name):
        """Collapse filename variants like bear_06_1.png and bear_12_1.png to one key."""
        if stimulus_name is None:
            return ""
        stim = os.path.basename(str(stimulus_name).strip())
        stem, _ = os.path.splitext(stim)
        # Remove trailing variant suffixes "_<num>_<num>".
        m = re.match(r"^(.*?)(?:_\d+_\d+)$", stem)
        if m and m.group(1):
            return m.group(1)
        # Fallback: strip a single numeric suffix if present.
        m2 = re.match(r"^(.*?)(?:_\d+)$", stem)
        if m2 and m2.group(1):
            return m2.group(1)
        return stem

    def _build_summary(self, trial_df, score_thr, rate_thr, min_trials):
        if trial_df.empty or "stimulus" not in trial_df.columns:
            return pd.DataFrame()

        try:
            grouping_idx = int(self.cmb_grouping.currentIndex())
        except Exception:
            grouping_idx = 0
        if grouping_idx == 1 and "stimulus_group" in trial_df.columns:
            group_col = "stimulus_group"
        elif grouping_idx == 2:
            group_col = "stimulus"
        else:
            group_col = "canonical_stimulus" if "canonical_stimulus" in trial_df.columns else "stimulus"
        grp = trial_df.groupby(group_col, sort=False)
        if "trial_responsive" not in trial_df.columns:
            trial_df = trial_df.copy()
            if "trial_score" in trial_df.columns:
                trial_df["trial_responsive"] = np.asarray(trial_df["trial_score"], dtype=float) >= score_thr
            else:
                trial_df["trial_responsive"] = False
            grp = trial_df.groupby(group_col, sort=False)

        summary = grp.agg(
            n_trials=("trial_index", "count"),
            first_trial=("trial", "min"),
            mean_score=("trial_score", "mean"),
            median_score=("trial_score", "median"),
            max_score=("trial_score", "max"),
            responsive_trials=("trial_responsive", lambda x: int(np.sum(np.asarray(x, dtype=bool)))),
        ).reset_index()
        summary = summary.rename(columns={group_col: "stimulus"})

        if group_col == "canonical_stimulus":
            variant_count = (
                trial_df.groupby(group_col)["stimulus"]
                .nunique(dropna=True)
                .rename("n_variants")
                .reset_index()
                .rename(columns={group_col: "stimulus"})
            )
            variants_preview = (
                trial_df.groupby(group_col)["stimulus"]
                .agg(lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)})[:4]))
                .rename("variants_preview")
                .reset_index()
                .rename(columns={group_col: "stimulus"})
            )
            summary = summary.merge(variant_count, on="stimulus", how="left")
            summary = summary.merge(variants_preview, on="stimulus", how="left")
        elif group_col == "stimulus_group":
            member_count = (
                trial_df.groupby(group_col)["canonical_stimulus"]
                .nunique(dropna=True)
                .rename("n_canonical_stimuli")
                .reset_index()
                .rename(columns={group_col: "stimulus"})
            )
            members_preview = (
                trial_df.groupby(group_col)["canonical_stimulus"]
                .agg(lambda x: ", ".join(sorted({str(v) for v in x if pd.notna(v)})[:6]))
                .rename("stimuli_preview")
                .reset_index()
                .rename(columns={group_col: "stimulus"})
            )
            summary = summary.merge(member_count, on="stimulus", how="left")
            summary = summary.merge(members_preview, on="stimulus", how="left")

        if "responsive_channels_list" in trial_df.columns:
            chan_union = (
                trial_df.groupby(group_col)["responsive_channels_list"]
                .agg(
                    lambda rows: sorted(
                        {
                            str(ch)
                            for row in rows
                            if isinstance(row, (list, tuple, set))
                            for ch in row
                        }
                    )
                )
                .reset_index()
                .rename(columns={group_col: "stimulus", "responsive_channels_list": "_responsive_channels_union"})
            )
            if "_responsive_channels_union" in chan_union.columns:
                chan_union["n_responsive_channels"] = chan_union["_responsive_channels_union"].map(len)
                chan_union["responsive_channels"] = chan_union["_responsive_channels_union"].map(
                    lambda lst: ", ".join(lst)
                )
                chan_union = chan_union.drop(columns=["_responsive_channels_union"])
                summary = summary.merge(chan_union, on="stimulus", how="left")

        summary["responsive_rate_pct"] = (
            100.0 * summary["responsive_trials"] / np.maximum(summary["n_trials"], 1)
        )
        summary["classification"] = np.where(
            (summary["n_trials"] >= int(min_trials))
            & (summary["responsive_rate_pct"] >= float(rate_thr)),
            "responsive",
            "non-responsive",
        )
        summary = summary.sort_values(
            by=["classification", "responsive_rate_pct", "mean_score", "n_trials"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        return summary

    def recompute_summary(self):
        if self._trial_df_raw.empty:
            self.clear("No trial data available for summary.")
            return

        score_thr = float(self.spb_score_thr.value())
        rate_thr = float(self.spb_rate_thr.value())
        min_trials = int(self.spb_min_trials.value())

        trial_df = self._trial_df_raw.copy()
        if "channel_delta_pairs" in trial_df.columns:
            responsive_lists = []
            responsive_counts = []
            trial_scores = []
            for pairs in trial_df["channel_delta_pairs"].tolist():
                if not isinstance(pairs, (list, tuple)):
                    pairs = []
                valid = []
                for item in pairs:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    ch = str(item[0])
                    try:
                        val = float(item[1])
                    except Exception:
                        continue
                    if not np.isfinite(val):
                        continue
                    valid.append((ch, val))
                if valid:
                    trial_scores.append(float(max(v for _, v in valid)))
                    resp = sorted({ch for ch, v in valid if v >= score_thr})
                else:
                    trial_scores.append(np.nan)
                    resp = []
                responsive_lists.append(resp)
                responsive_counts.append(int(len(resp)))
            trial_df["trial_score"] = pd.to_numeric(trial_scores, errors="coerce")
            trial_df["responsive_channels_list"] = responsive_lists
            trial_df["responsive_channels"] = [", ".join(lst) for lst in responsive_lists]
            trial_df["n_responsive_channels"] = responsive_counts
            trial_df["trial_responsive"] = np.asarray(responsive_counts, dtype=int) > 0
        elif "trial_score" in trial_df.columns:
            trial_df["trial_responsive"] = np.asarray(trial_df["trial_score"], dtype=float) >= score_thr
            trial_df["responsive_channels_list"] = [[] for _ in range(len(trial_df))]
            trial_df["responsive_channels"] = ["" for _ in range(len(trial_df))]
            trial_df["n_responsive_channels"] = 0
        else:
            trial_df["trial_responsive"] = False
            trial_df["responsive_channels_list"] = [[] for _ in range(len(trial_df))]
            trial_df["responsive_channels"] = ["" for _ in range(len(trial_df))]
            trial_df["n_responsive_channels"] = 0

        self._summary_df = self._build_summary(
            trial_df=trial_df,
            score_thr=score_thr,
            rate_thr=rate_thr,
            min_trials=min_trials,
        )
        self._trial_df_current = trial_df.copy()

        trial_cols = [
            c
            for c in [
                "trial_index",
                "trial_index_in_file",
                "run_id",
                "block_id",
                "source_fif",
                "trial",
                "canonical_stimulus",
                "stimulus_group",
                "stimulus",
                "onset_time",
                "offset_time",
                "onset_meg_s",
                "offset_meg_s",
                "duration",
                "trial_score",
                "n_responsive_channels",
                "responsive_channels",
                "trial_responsive",
                "timing_delta_s",
                "top_response_channel",
                "top_response_score",
            ]
            if c in trial_df.columns
        ]
        display_trials = trial_df[trial_cols].copy()
        if "trial_responsive" in display_trials.columns:
            display_trials["trial_responsive"] = display_trials["trial_responsive"].map(
                {True: "yes", False: "no"}
            )

        summary_cols = [
            c
            for c in [
                "stimulus",
                "n_trials",
                "first_trial",
                "n_variants",
                "variants_preview",
                "n_canonical_stimuli",
                "stimuli_preview",
                "n_responsive_channels",
                "responsive_channels",
                "responsive_trials",
                "responsive_rate_pct",
                "mean_score",
                "median_score",
                "max_score",
                "classification",
            ]
            if c in self._summary_df.columns
        ]
        display_summary = self._summary_df[summary_cols].copy()

        self.trials_model = PandasModel(display_trials)
        self.tbl_trials.setModel(self.trials_model)
        self.tbl_trials.resizeColumnsToContents()

        self.summary_model = PandasModel(display_summary)
        self.tbl_summary.setModel(self.summary_model)
        self.tbl_summary.resizeColumnsToContents()

        n_stim = int(len(display_summary))
        n_resp = int(np.sum(display_summary["classification"] == "responsive")) if "classification" in display_summary.columns else 0
        n_non = max(0, n_stim - n_resp)
        grouping_label = "canonical stimulus"
        try:
            grouping_label = str(self.cmb_grouping.currentText() or grouping_label).lower()
        except Exception:
            pass
        block_id = self._metadata.get("block_id", "n/a")
        run_id = self._metadata.get("run_id", "n/a")
        try:
            n_fif = int(self._metadata.get("n_fif_processed", 1))
        except Exception:
            n_fif = 1
        if n_fif > 1:
            self.status_label.setText(
                f"{n_fif} FIF blocks: {n_resp} responsive, {n_non} non-responsive entries "
                f"({grouping_label}; any-channel threshold={score_thr:.3f}, "
                f"rate>={rate_thr:.1f}%, min_trials={min_trials})"
            )
        else:
            self.status_label.setText(
                f"Run {run_id}, block {block_id}: {n_resp} responsive, {n_non} non-responsive entries "
                f"({grouping_label}; any-channel threshold={score_thr:.3f}, "
                f"rate>={rate_thr:.1f}%, min_trials={min_trials})"
            )

    def get_summary_df(self):
        return self._summary_df.copy()

    def _trial_table_for_export(self):
        trial_df = self._trial_df_raw.copy()
        if trial_df.empty:
            trial_df = self._trial_df_current.copy()
        if trial_df.empty:
            return pd.DataFrame()

        if "stimulus" in trial_df.columns:
            if "canonical_stimulus" not in trial_df.columns:
                trial_df["canonical_stimulus"] = trial_df["stimulus"].map(self._canonicalize_stimulus_label)
            else:
                fallback = trial_df["stimulus"].map(self._canonicalize_stimulus_label)
                trial_df["canonical_stimulus"] = trial_df["canonical_stimulus"].fillna(fallback)

        preferred_cols = [
            "trial_index",
            "trial_index_in_file",
            "run_id",
            "block_id",
            "session_id",
            "source_fif",
            "source_fif_path",
            "source_original_fif_path",
            "source_ica_applied",
            "onset_sample",
            "offset_sample",
            "onset_meg_s",
            "offset_meg_s",
            "onset_time",
            "offset_time",
            "duration",
            "timing_delta_s",
            "stimulus",
            "canonical_stimulus",
            "stimulus_group",
            "concept",
            "trial_score",
            "baseline_rms_median",
            "response_rms_median",
            "top_response_channel",
            "top_response_score",
        ]
        excluded_cols = {"channel_delta_pairs", "responsive_channels_list"}
        ordered_cols = [col for col in preferred_cols if col in trial_df.columns]
        remaining_cols = [
            col
            for col in trial_df.columns
            if col not in ordered_cols and col not in excluded_cols
        ]
        return trial_df[ordered_cols + remaining_cols].copy()

    @staticmethod
    def _summary_stimulus_names(summary_df):
        if not isinstance(summary_df, pd.DataFrame) or summary_df.empty or "stimulus" not in summary_df.columns:
            return []
        names = []
        seen = set()
        for value in summary_df["stimulus"].tolist():
            if pd.isna(value):
                continue
            text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()
            if not text or text.lower() in {"unknown", "nan", "none", "null"} or text in seen:
                continue
            seen.add(text)
            names.append(text)
        return names

    def _image_references_by_summary_label(self, names):
        trial_df = self._trial_df_raw
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            return {str(name): [] for name in names}
        try:
            grouping_idx = int(self.cmb_grouping.currentIndex())
        except Exception:
            grouping_idx = 0
        if grouping_idx == 1 and "stimulus_group" in trial_df.columns:
            group_col = "stimulus_group"
        elif grouping_idx == 2:
            group_col = "stimulus"
        else:
            group_col = (
                "canonical_stimulus"
                if "canonical_stimulus" in trial_df.columns
                else "stimulus"
            )
        requested = {str(name): [] for name in names}
        if group_col not in trial_df.columns or "stimulus" not in trial_df.columns:
            return requested
        for _, row in trial_df.iterrows():
            label_value = row.get(group_col)
            stimulus_value = row.get("stimulus")
            if pd.isna(label_value) or pd.isna(stimulus_value):
                continue
            label = " ".join(str(label_value).split()).strip()
            stimulus = str(stimulus_value).strip()
            if label in requested and stimulus and stimulus not in requested[label]:
                requested[label].append(stimulus)
        return requested

    def create_embeddings_from_summary(self):
        names = self._summary_stimulus_names(self._summary_df)
        if not names:
            self.status_label.setText("No embeddings created: Stimulus Summary has no usable stimulus values.")
            return

        if self._embedding_thread is not None:
            self.status_label.setText("Embedding creation is already running.")
            return

        subject = str(self._metadata.get("subject_id", "subject"))
        run_id = str(self._metadata.get("run_id", "run"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"concept_embeddings_{subject}_run{run_id}_{ts}.json"
        default_dir = str(self._metadata.get("events_dir", os.getcwd()))
        default_path = os.path.join(default_dir, default_name)
        grouping_label = str(self.cmb_grouping.currentText() or "Summary labels")
        current_trial_df = (
            self._trial_df_current.copy()
            if not self._trial_df_current.empty
            else self._trial_df_raw.copy()
        )
        dialog = _EmbeddingCreationDialog(
            names=names,
            grouping_label=grouping_label,
            default_path=default_path,
            parent=self.widget,
            trial_df=current_trial_df,
        )
        if not dialog.exec_():
            return
        config = dialog.configuration()
        out_path = str(config.pop("output_path"))
        config["source_grouping"] = grouping_label
        config["trial_df"] = current_trial_df

        provider = str(config.get("provider", "openai") or "openai")
        if provider in {"clip_image", "dinov2_image"}:
            config["image_references_by_label"] = self._image_references_by_summary_label(
                names
            )
        api_key = ""
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                api_key, ok = QtWidgets.QInputDialog.getText(
                    self.widget,
                    "OpenAI API key",
                    "OPENAI_API_KEY is not set. Enter an OpenAI API key for this embedding request:",
                    QtWidgets.QLineEdit.Password,
                )
                api_key = str(api_key or "").strip()
                if not ok or not api_key:
                    self.status_label.setText("No embeddings created: OpenAI API key was not provided.")
                    return

        self.btn_create_embeddings.setEnabled(False)
        if provider == "word2vec":
            generator_label = f"Word2Vec ({os.path.basename(config['word2vec_model_path'])})"
        elif provider in {"clip_image", "clip_text", "dinov2_image"}:
            generator_label = str(config.get("transformer_model", provider))
        else:
            generator_label = str(config.get("openai_model", "OpenAI"))
        self.status_label.setText(f"Creating {generator_label} vectors for {len(names)} summary labels...")
        thread = QtCore.QThread()
        worker = _ConceptEmbeddingWorker(
            names=names,
            output_path=out_path,
            config=config,
            api_key=api_key,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_embeddings_created)
        worker.failed.connect(self._on_embeddings_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(lambda *_args: worker.deleteLater())
        worker.failed.connect(lambda *_args: worker.deleteLater())
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_embedding_worker_refs)
        self._embedding_thread = thread
        self._embedding_worker = worker
        thread.start()

    def _clear_embedding_worker_refs(self):
        self._embedding_thread = None
        self._embedding_worker = None

    def _on_embeddings_created(self, output_path, n_names, dim, details=""):
        self.btn_create_embeddings.setEnabled(True)
        detail_suffix = f"; {details}" if details else ""
        self.status_label.setText(
            f"Saved {int(n_names)} concept embeddings ({int(dim)}D) to {output_path}{detail_suffix}"
        )

    def _on_embeddings_failed(self, message):
        self.btn_create_embeddings.setEnabled(True)
        self.status_label.setText(f"Embedding creation failed: {message}")

    def export_report(self):
        if self._summary_df.empty:
            self.status_label.setText("Nothing to export: summary table is empty.")
            return

        subject = str(self._metadata.get("subject_id", "subject"))
        run_id = str(self._metadata.get("run_id", "run"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"stimulus_response_report_{subject}_run{run_id}_{ts}.csv"
        default_dir = str(self._metadata.get("events_dir", os.getcwd()))
        default_path = os.path.join(default_dir, default_name)
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save stimulus response summary",
            default_path,
            "CSV files (*.csv);;All files (*.*)",
        )
        if not out_path:
            return

        try:
            summary_df = self._summary_df.copy()
            trial_df = self._trial_df_current.copy() if not self._trial_df_current.empty else self._trial_df_raw.copy()
            if not out_path.lower().endswith(".csv"):
                out_path = f"{out_path}.csv"
            trial_path = out_path[:-4] + "_trials.csv"
            summary_df.to_csv(out_path, index=False)
            trial_df.to_csv(trial_path, index=False)
            self.status_label.setText(
                f"Saved summary to {out_path} and trials to {trial_path}"
            )
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")

    def export_trial_table(self):
        trial_df = self._trial_table_for_export()
        if trial_df.empty:
            self.status_label.setText("Nothing to export: mapped trial table is empty.")
            return

        missing_required = [col for col in ("source_fif_path", "onset_sample") if col not in trial_df.columns]
        label_cols = {"stimulus", "canonical_stimulus", "stimulus_group", "concept", "concept_label"}
        has_label = any(col in trial_df.columns for col in label_cols)
        if missing_required or not has_label:
            parts = []
            if missing_required:
                parts.append("missing " + ", ".join(missing_required))
            if not has_label:
                parts.append("missing a stimulus/concept label column")
            self.status_label.setText("Cannot export CLI trial table: " + "; ".join(parts) + ".")
            return

        subject = str(self._metadata.get("subject_id", "subject"))
        run_id = str(self._metadata.get("run_id", "run"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"concept_decoding_trial_table_{subject}_run{run_id}_{ts}.csv"
        default_dir = str(self._metadata.get("events_dir", os.getcwd()))
        default_path = os.path.join(default_dir, default_name)
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save concept decoding trial table",
            default_path,
            "CSV files (*.csv);;All files (*.*)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".csv"):
            out_path = f"{out_path}.csv"

        try:
            trial_df.to_csv(out_path, index=False)
            self.status_label.setText(
                f"Saved CLI-ready trial table ({len(trial_df)} trials) to {out_path}"
            )
        except Exception as exc:
            self.status_label.setText(f"Trial table export failed: {exc}")

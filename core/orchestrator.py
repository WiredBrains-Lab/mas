"""
Experiment orchestrator
=======================

Coordinates MEG preprocessing, visualization, and analysis across the
application. Wires Qt signals between the MEG UI panels and manages the
currently loaded FIF data.

Notes
-----
- UI components (SignalViewer, PowerSpectrum, MEG analysis, and 3D) are
  constructed and connected here.
"""
# Author: Sunil Mathew
# Date: June 14, 2026
# Orchestrator class for the MEG analysis suite.

import glob
import json
import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd
from datetime import datetime
import mne
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QObject, Signal

from pyqtgraph.dockarea.Dock import Dock

import sys
import traceback

sys.path.append(os.path.dirname(__file__))
from viewer3d.meg_3d import MEG3DViewer
from dsp.filtering import Filtering
from core.meg_signal_viewer import MEGSignalViewer
from dsp.power_spectrum import PowerSpectrum
from core.meg_analysis_panel import MEGAnalysisPanel

from core.fif_processing import FIFProcessingMixin
from core.concept_decoding_signal import compute_signal_space_concept_decoding
from core.concept_decoding_source import (
    build_source_parcel_trial_time_courses,
    clear_parcel_timecourse_cache,
    compute_source_space_concept_decoding,
)
from core.meg_channels import emit_bad_meg_channel_log, good_meg_channel_names, good_meg_picks


FIF_EVENTS_CHANNEL_LABEL = "events"


class Communicate(QObject):
    """Qt signals used by the MEG application components."""
    log = Signal(str)
    progress = Signal(int, str, bool)  # progress percentage, message, b_append
    channel_waveform_clicked = Signal(int)
    sig_view_param = Signal(str, object)
    sig_view_filt_param = Signal(str, object)
    sig_view_nav = Signal(str)
    sig_view_go_to = Signal(datetime)
    sig_view_acq = Signal(bool)
    sig_view_acq_channels = Signal(dict)
    sig_view_acq_channels_selection = Signal(dict)
    start_all_ch_psd_acq = Signal()
    stop_all_ch_psd_acq = Signal()

class Orchestrator(FIFProcessingMixin):
    """High-level coordinator for acquisition, processing, and UI.

    Parameters
    ----------
    sig_view : pyqtgraph.dockarea.DockArea or None
        Root dock area used to attach views when UI is enabled.
    ui : bool, default True
        When False, constructs without UI wiring (headless mode).
    """
    def __init__(self, sig_view=None, ui=True) -> None:
        np.random.seed(123)
        self.has_ui = ui
        self.sig_view = sig_view
        self.c = Communicate()
        self.asc = True
        self.b_recording = False
        
        self.init_signal_params()
        self.init_ui_layout()
        self.init_comm()

    def __del__(self):
        """Cleanup when orchestrator is destroyed"""
        try:
            if hasattr(self, 'meg_3d') and self.meg_3d is not None:
                self.meg_3d._teardown_renderer()
        except:
            pass

    #region Communication

    def init_comm(self):
        """Connect signals used by the MEG-only application."""
        self.c.log.connect(self.update_log)
        self.c.sig_view_param.connect(self.update_sig_view_param)
        self.c.sig_view_filt_param.connect(self.update_sig_view_filt_param)
        self.c.sig_view_nav.connect(self.nav_sig_view)
        self.c.sig_view_go_to.connect(self.go_to_time)
        self.c.sig_view_acq_channels.connect(self.signal_viewer.update_channel_list)
        self.c.sig_view_acq_channels_selection.connect(self.update_channel_selection)



    def load_study(self, folder_path):
        """Load study metadata, NSx data, and initialize UI state.

        Parameters
        ----------
        folder_path : str
            Study root directory containing concepts, NSx files, and results.
        """
        # Check if a task is currently running
        task_running = False
        if hasattr(self, 'b_recording') and self.b_recording:
            task_running = True
        elif hasattr(self.tasks, 'b_task_started') and self.tasks.b_task_started:
            task_running = True
        elif hasattr(self, 'b_start_acq') and self.b_start_acq:
            task_running = True
        elif hasattr(self, 'b_start_seq_acq') and self.b_start_seq_acq:
            task_running = True
            
        if task_running:
            task_running_msg_box = QtWidgets.QMessageBox()
            task_running_msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            task_running_msg_box.setText("Cannot load study while a task is currently running.")
            task_running_msg_box.setInformativeText("Please stop the current task before loading a study.")
            task_running_msg_box.setWindowTitle("Task Currently Running")
            task_running_msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            task_running_msg_box.exec()
            return

        # Locals
        b_legacy = False
        try:
            if not os.path.isdir(folder_path):
                self.c.log.emit(f'Error: {folder_path} is not a valid directory')
                return
            
            # search for first *.ns5 file
            ns_file_path = glob.glob(os.path.join(folder_path, '*.ns5'))[0] if len(glob.glob(os.path.join(folder_path, '*.ns5'))) > 0 else None
            if ns_file_path:
                b_legacy = self.read_ripple_nsx_file(filename=ns_file_path)
            
            # Check if task_info.json exists
            task_info_path = os.path.join(folder_path, 'task_info.json')
            if not b_legacy and os.path.exists(task_info_path):
                self.concepts_studio.load_img_info_table(folder_path=folder_path)
                self.responses_list.load_responses_events_tables(load_folder=folder_path)

                self.clustering.clus_info_dict = self.responses_list.clus_info_dict
                micro_ch_lbls = list(self.clustering.clus_info_dict.keys())
                self.clustering.update_clustering_cmb_channels(micro_ch_lbls)
                self.clustering.plot_all_ch_templates()
                self.tasks.load_screening_config(folder_path=folder_path)
                
                # Update UI to reflect review mode
                self.update_ui_for_review_mode()

                self.c.log.emit(f'Loading study data from {folder_path}')
            elif b_legacy:
                self.c.log.emit(f'Loading legacy study data from {folder_path}')
                
                # Update UI to reflect review mode
                self.update_ui_for_review_mode()
            else:
                self.c.log.emit(f'Error: task_info.json not found in {folder_path}. '
                                f'This must be a legacy study. Make sure you ran grapes_debug.m '
                                f'to generate files to be able to open the study in MAS.')
            
        except:
            print(traceback.format_exc())
            self.c.log.emit(f'Error loading study data from {folder_path}')

    def clear_review_mode_data(self):
        """Clear data and UI state when exiting review mode to start a new task."""
        try:
            # Reset responses_list state
            if hasattr(self.responses_list, 'b_review_mode'):
                self.responses_list.b_review_mode = False
            if hasattr(self.responses_list, 'b_legacy'):
                self.responses_list.b_legacy = False
            
            # Clear study-related data
            if hasattr(self.responses_list, 'image_names'):
                self.responses_list.image_names = []
            if hasattr(self.responses_list, 'img_paths'):
                self.responses_list.img_paths = []
            if hasattr(self.responses_list, 'study_folder'):
                self.responses_list.study_folder = None
            if hasattr(self.responses_list, 'pics_order_1d'):
                self.responses_list.pics_order_1d = None
            if hasattr(self.responses_list, 'ranking_table'):
                self.responses_list.ranking_table = None
            if hasattr(self.responses_list, 'pics_onset'):
                self.responses_list.pics_onset = None
            if hasattr(self.responses_list, 'used_pics_names'):
                self.responses_list.used_pics_names = None
            if hasattr(self.responses_list, 'stimulus_list'):
                self.responses_list.stimulus_list = []
            if hasattr(self.responses_list, 'ifr'):
                self.responses_list.ifr = None
                
            # Clear UI tables that may show loaded data
            if hasattr(self.responses_list, 'responses_table') and hasattr(self.responses_list.responses_table, 'setModel'):
                self.responses_list.responses_table.setModel(None)
            
            # Clear clustering data that may be from loaded study
            if hasattr(self.clustering, 'clus_info_dict'):
                self.clustering.clus_info_dict = {}
            
            # Reset task label
            if hasattr(self.responses_list, 'lbl_task_info'):
                self.responses_list.lbl_task_info.setText('No task running')
            
            # Re-enable concepts studio controls
            if hasattr(self.concepts_studio, 'enable_concepts_controls'):
                self.concepts_studio.enable_concepts_controls(bEnable=True)
                
            # Clear any loaded study info from concepts studio
            if hasattr(self.concepts_studio, 'task_info'):
                self.concepts_studio.task_info = None
                
            # Reset task folder path in concepts studio if it was set from loaded study
            if hasattr(self.concepts_studio, 'task_folder'):
                self.concepts_studio.task_folder = None
                
            # Reset any sequence numbers or task state from loaded data
            if hasattr(self.tasks, 'seq_no'):
                self.tasks.seq_no = 0
                
        except Exception as e:
            print(f"Error clearing review mode data: {e}")
            print(traceback.format_exc())

    def update_ui_for_review_mode(self):
        """Update UI elements to reflect that we're in review mode."""
        try:                
            # Disable concepts studio controls since we're reviewing loaded data
            if hasattr(self.concepts_studio, 'enable_concepts_controls'):
                self.concepts_studio.enable_concepts_controls(bEnable=False)
                
        except Exception as e:
            print(f"Error updating UI for review mode: {e}")
            print(traceback.format_exc())

    def experiment_ended(self):
        self.acq_send_queue.put({'experiment_end': {}})
        
        
        # Clean up shared memory buffer
        if self.shared_vis_buffer:
            try:
                self.shared_vis_buffer = None
                print("Cleaned up shared buffer reference in orchestrator")
            except Exception as e:
                print(f"Error cleaning up shared buffer reference: {e}")
        
        # Clean up event-driven queue workers
        # try:
        #     self.queue_manager.stop_all()
        #     print("Stopped all queue workers")
        # except Exception as e:
        #     print(f"Error stopping queue workers: {e}")
        
        # Note: Backup is now handled after spike sorting completion
        # to ensure all processing is complete before backup

    def start_backup_process(self):
        """Start backup after spike sorting completes and UI is settled."""
        self.responses_list.save_responses_events_tables(save_folder=self.tasks.task_folder)
        self.concepts_studio.save_task_info()
        self.concepts_studio.enable_concepts_controls(bEnable=True)
        self.signal_viewer.btnAcqStart.setEnabled(True)

        if not self.tasks.backup:
            return
            
        if not hasattr(self.tasks, 'task_folder') or not self.tasks.task_folder:
            self.c.log.emit("Cannot start backup: No task folder available")
            return
            
        try:
            self.c.log.emit("Starting backup process after spike sorting completion...")
            # start backup process
            process, queue = start_backup(local_source=self.tasks.task_folder, 
                                          network_path=config['backup_path'], 
                                          username=config['backup_user'], 
                                          password=config['backup_password'])

            bOK = True
            # --- Monitor progress ---
            while process.is_alive() or not queue.empty():
                while not queue.empty():
                    msg_type, msg, txt = queue.get()
                    if msg_type == "progress":
                        # print(f"Backup Progress: {msg:.1f}%")
                        self.c.progress.emit(int(msg), txt, False)
                    elif msg_type == "info":
                        print(f"Backup Info: {msg}")
                        self.c.log.emit(f"Backup: {msg}")
                    elif msg_type == "error":
                        print(f"Backup Error: {msg}")
                        self.c.log.emit(f"Backup Error: {msg}")
                        bOK = False
            process.join()
            if bOK:
                backed_up_path = os.path.join(config['backup_path'], os.path.basename(self.tasks.task_folder))
                self.c.log.emit(f"Backup to {backed_up_path} completed successfully.")
        except Exception as e:
            print(f"Error during backup: {e}")
            self.c.log.emit(f"Backup failed: {e}")
            print(traceback.format_exc())


    def create_next_subscreening_config(self, subscr_idx, update_table=True):
        scr_selected_stims, scr_removed_stims, responsive_stims = self.responses_list.get_scr_shown_stims(subscr_idx-1) # selected in last subscreening
        prev_subscreening_config = self.tasks.get_subscreening_config(subscr_idx-1)
        nxt_scr_stim_list, extra_stims, removed_stims = self.concepts_studio.create_subscreening_config(scr_id=subscr_idx, 
                                                        scr_selected_stims=scr_selected_stims, 
                                                        discarded_stimuli=scr_removed_stims,
                                                        responsive_stims=responsive_stims,
                                                        prev_subscreening_config=prev_subscreening_config,
                                                        update_table=update_table)
        if hasattr(self.responses_list, 'nxt_scr_stims'):
            self.responses_list.nxt_scr_stims = nxt_scr_stim_list
            self.responses_list.endangered_stims = extra_stims
            self.responses_list.removed_stims = removed_stims

        

    def update_scr_config(self, subscr_idx, scr_config, start_subscreening=False):
        self.tasks.update_scr_config(subscr_idx, scr_config, start_subscreening)
        self.spike_sorting_send_queue.put({'curr_subscr_sort_seq_id': scr_config[f'subscreening_{subscr_idx}'].sort_seq_id})

    def show_stim_img(self):
        self.tasks.show_stim_img()

    def hide_stim_img(self):
        self.tasks.hide_stim_img()

    def update_exp_config(self, config):
        self.concepts_studio.update_exp_config(config)

    def update_task_folder(self, task_folder):
        self.concepts_studio.task_folder = task_folder
        self.responses_list.task_folder = task_folder
        # Provide task folder to filtering so it can locate MATLAB notch files
        try:
            if hasattr(self, 'filtering') and hasattr(self.filtering, 'set_task_folder'):
                self.filtering.set_task_folder(task_folder)
        except Exception as e:
            print(f"Warning: could not update filtering task folder: {e}")

    def update_related_concepts_single(self):
        self.concepts_studio.update_related_concepts(b_llm=True)

    def update_related_concepts(self):
        self.concepts_studio.post_process_related_concepts()

    def show_msg_box(self, title, msg):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText(msg)
        msg_box.setWindowTitle(title)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        returnValue = msg_box.exec()
        return returnValue

    def start_task(self, state):
        # Check if in review mode and ask for confirmation to exit
        if hasattr(self.responses_list, 'b_review_mode') and self.responses_list.b_review_mode:
            review_msg_box = QtWidgets.QMessageBox()
            review_msg_box.setIcon(QtWidgets.QMessageBox.Question)
            review_msg_box.setText("You are currently in review mode (viewing previously collected data). " \
                                   "Starting a new task will exit review mode and begin a new experiment.")
            review_msg_box.setInformativeText("Do you want to exit review mode and start a new task?")
            review_msg_box.setWindowTitle("Exit Review Mode")
            review_msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            review_msg_box.setDefaultButton(QtWidgets.QMessageBox.No)
            
            returnValue = review_msg_box.exec()
            if returnValue == QtWidgets.QMessageBox.No:
                return  # User cancelled, don't start the task
            else:
                # User confirmed, exit review mode
                self.clear_review_mode_data()
                self.c.log.emit("Exited review mode to start new task")

        bOK = self.tasks.b_ripple_init
        if not bOK:
            # pop up a message box
            trellis_msg_box = QtWidgets.QMessageBox()
            trellis_msg_box.setIcon(QtWidgets.QMessageBox.Information)
            trellis_msg_box.setText("Trellis is not running. " \
                                    "Please start Trellis and try again "
                                    "or click OK to use digital simulator.")
            trellis_msg_box.setWindowTitle("Trellis not running")
            trellis_msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            # msgBox.buttonClicked.connect(msgButtonClick)

            returnValue = trellis_msg_box.exec()
            if returnValue == QtWidgets.QMessageBox.Ok:
                self.tasks.set_dig_sim(dig_sim=True)
                self.c.log.emit("Using digital simulator for acquisition. There will be no recording.")
                bOK = True
            else:
                bOK = False
                self.acq_send_queue.put({'init_ripple': {}})
                return


        bOK = self.concepts_studio.subject_id is not None       


        if not bOK:
            self.show_msg_box("Subject ID not selected", 
                              "Select a subject from Concepts Studio combobox before starting the task.")
            return
        
        bOK = self.tasks.priority_bundles != []
        if not bOK:
            priority_msg_box = QtWidgets.QMessageBox()
            priority_msg_box.setIcon(QtWidgets.QMessageBox.Information)
            priority_msg_box.setText("No priority channels selected. "\
                                     "Select priority channels by clicking items in the Priority Channels combobox. Click OK to continue without priority channels.")
            priority_msg_box.setWindowTitle("No priority channels selected")
            priority_msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            returnValue = priority_msg_box.exec()
            if returnValue == QtWidgets.QMessageBox.Ok:
                bOK = True
                print("Continuing without priority channels.")
            else:
                bOK = False
                return
        
        
        if bOK:
            if self.tasks.task_name.startswith('gaps'):
                self.tasks.start_gaps_task(subject_id=self.concepts_studio.subject_id)
                return
            bOK = self.concepts_studio.scr_config['subscreening_1'].task_name == self.tasks.task_name
            if bOK:
                self.tasks.start_rsvp_task(subject_id=self.concepts_studio.subject_id)
                self.concepts_studio.enable_concepts_controls(bEnable=False)
                self.responses_list.concepts_studio = self.concepts_studio
            else:
                self.show_msg_box("Task name mismatch", 
                                  "The task name in Concepts Studio and Tasks do not match. Please ensure they are the same.")
            

    def update_trellis_status(self, state):
        self.tasks.b_trellis = state

    def start_stop_acquisition(self, state):
        # Check if in review mode and starting acquisition
        if state and hasattr(self.responses_list, 'b_review_mode') and self.responses_list.b_review_mode:
            acq_review_msg_box = QtWidgets.QMessageBox()
            acq_review_msg_box.setIcon(QtWidgets.QMessageBox.Question)
            acq_review_msg_box.setText("You are currently in review mode (viewing recorded data).")
            acq_review_msg_box.setInformativeText("Starting acquisition will exit review mode and switch to live neural signal monitoring.\n\n" +
                                                 "Do you want to exit review mode and start live acquisition?")
            acq_review_msg_box.setWindowTitle("Switch to Live Acquisition")
            acq_review_msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            acq_review_msg_box.setDefaultButton(QtWidgets.QMessageBox.No)
            
            returnValue = acq_review_msg_box.exec()
            if returnValue == QtWidgets.QMessageBox.No:
                return  # User cancelled, don't start acquisition
            else:
                # User confirmed, exit review mode
                self.clear_review_mode_data()
                self.c.log.emit("Exited review mode to start live acquisition")

        if state:
            # bOK = self.ripple_acq.start_acq()
            self.acq_send_queue.put({'start_acq': True})
            self.signal_viewer.update_acq_btn(acq=True)
        else:
            # bOK = self.ripple_acq.stop_acq()
            self.acq_send_queue.put({'stop_acq': {}})

        self.b_start_acq = state

    def start_stop_task_acquisition(self, state, subscr_idx=None, 
                                    subscr_seq_idx=None, subscr_seq_count=None,
                                    seq_img_names=None):
        if state:
            self.acq_send_queue.put({'sequence_begin': {
                'subscr_id': subscr_idx,
                'seq_id': subscr_seq_idx,
                'subscr_seq_count': subscr_seq_count,
                'seq_img_names': seq_img_names
            }})
            self.b_start_seq_acq = True
        else:
            self.acq_send_queue.put({'sequence_end': {'seq_img_names': seq_img_names}})
            self.b_start_seq_acq = False

        if self.b_recording:
            # disable the acquisition button
            self.signal_viewer.btnAcqStart.setEnabled(False)
        else:
            self.signal_viewer.btnAcqStart.setEnabled(True)

        self.b_start_acq = state
        

    def start_or_stop_recording(self, state):
        if state:
            self.ripple_acq.start_recording(task_id=self.tasks.task_id)
        else:
            self.ripple_acq.stop_recording()
            self.move_recorded_files()

    def move_recorded_files(self, data_folder=None):
        files = os.listdir(data_folder)
        rec_files = [f for f in files if f.startswith(f'{self.tasks.task_id}')]
        for f in rec_files:
            try:
                shutil.move(os.path.join(data_folder, f), self.tasks.task_folder)
                self.c.log.emit(f'Moved {f} to {self.tasks.task_folder}')
            except:
                self.c.log.emit(f'Error moving {f} to {self.tasks.task_folder}')
                print(traceback.format_exc())

    def update_recording_status(self, state):
        self.b_recording = state
        self.tasks.update_recording_status(state)

    def update_subscreening_img_list(self, img_list=None):
        try:
            if img_list is None:
                return

            self.tasks.subscr_img_list = img_list
            self.c.log.emit(f'Starting RSVPSCR with {len(self.tasks.subscr_img_list)} images')
        except:
            print(traceback.format_exc())

    def update_subscreening_img(self, img_name, subscr_idx=None, seq_idx=None, seq_img_idx=None, trial_idx=None):
        try:
            # Add stimulus to responses display
            # self.responses_list.add_stimulus(id=id, img=img, lbl=os.path.basename(img_path))
            self.responses_list.update_stimulus_shown_order(lbl=img_name, subscr_id=subscr_idx, seq_id=seq_idx, seq_img_id=seq_img_idx, trial_id=trial_idx)
        except:
            print(traceback.format_exc())

    def nav_sig_view(self, direction='forward'):
        if self.b_fif_loaded:
            self.navigate_fif(direction=direction)

    def go_to_time(self, date_time):
        if self.b_fif_loaded:
            # FIF playback currently supports relative forward/back navigation.
            return

    def update_sig_view_param(self, param, value):
        setattr(self.signal_viewer, param, value)
        if param == 'channel':
            if self.b_fif_loaded:
                self.update_fif_signal_window()
        else:
            if self.b_fif_loaded:
                if param in {'length'}:
                    self.update_fif_signal_window()

    def update_task_clus_param(self, param, value):
        setattr(self.clustering, param, value)
        setattr(self.ripple_nsx, param, value)
        setattr(self.ripple_acq, param, value)

    def update_sig_view_filt_param(self, param, value):
        setattr(self.filtering, param, value)
        if self.b_fif_loaded:
            if param in {'ch_count', 'lowcut', 'highcut'}:
                self._limit_fif_selected_channels()
                self.update_fif_signal_window()
            if param in {'lowcut', 'highcut', 'filt_order', 'apply_notches'}:
                try:
                    self._update_psd_tab_from_fif()
                except Exception:
                    pass

    def channel_waveform_clicked(self, plt_idx=None):
        '''
        Called when a waveform in the signal viewer is clicked.
        Triggers power spectrum computation for the selected channel.
        '''
        if self.b_start_acq:
            # self.ripple_acq.channel_waveform_clicked(plt_idx=plt_idx)
            self.acq_send_queue.put({'channel_waveform_clicked': {'plt_idx': plt_idx}})
        elif self.b_nsx_read:
            self.ripple_nsx.channel_waveform_clicked(plt_idx=plt_idx)

    def update_channel_selection(self, channel_dict):
        if self.b_fif_loaded:
            self._update_fif_selected_channels_from_tree(channel_dict)
            self.update_fif_signal_window()

    def start_all_channel_psd_acquisition(self):
        """
        Start all-channel PSD data acquisition for notch calculation.
        This is triggered by the "Acquire" checkbox in power_spectrum.py.
        """
        if self.b_start_acq:
            self.acq_send_queue.put({'start_all_ch_psd_acq': {}})
            print("Triggered all-channel PSD acquisition")
        else:
            print("Cannot start all-channel PSD acquisition: acquisition not running")
    
    def stop_all_channel_psd_acquisition(self):
        """
        Stop all-channel PSD data acquisition.
        This is triggered by unchecking the "Acquire" checkbox in power_spectrum.py.
        """
        if self.b_start_acq:
            self.acq_send_queue.put({'stop_all_ch_psd_acq': {}})
            print("Triggered stop all-channel PSD acquisition")
        else:
            print("Cannot stop all-channel PSD acquisition: acquisition not running")    

    def update_log(self, message):
        """Append an application message to the MEG log dock."""
        print(message)
        if hasattr(self, 'txt_log'):
            self.txt_log.appendPlainText(str(message))
            scrollbar = self.txt_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    #endregion Communication
    
    #region Gamepad support

    def init_gamepad(self):
        """
        Initialize pygame and set up a QTimer to poll for gamepad events.
        """
        pygame.init()
        pygame.joystick.init()
        self.gamepad = None
        if pygame.joystick.get_count() > 0:
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            print(f"Gamepad connected: {self.gamepad.get_name()}")
        else:
            print("No gamepad connected.")
        self.gamepad_timer = QtCore.QTimer()
        self.gamepad_timer.timeout.connect(self.poll_gamepad)
        # self.gamepad_timer.start(50)  # Poll every 50 ms

    def poll_gamepad(self):
        """
        Poll for gamepad events and call continue_key_press_handler if 'A' is pressed.
        """
        if not self.gamepad:
            return
        try:
            pygame.event.pump()
            # Logitech F710 'A' button is usually button 0
            if self.gamepad.get_button(0):
                self.tasks.continue_key_press_handler()
        except Exception as e:
            print(f"Gamepad polling error: {e}")

    #endregion Gamepad support

    def init_signal_params(self):
        self.b_fif_loaded = False
        self._reset_fif_analysis_state()

        
    def init_ui_layout(self):
        """
        Initializes the layout of the signal viewer.
        """
        self.chs_view = Dock(name='Channels', closable=True)
        self.chs_view.sigClosed.connect(self.dock_closed_event_handler)
        self.log_view = Dock(name='Log', closable=True)
        self.log_view.sigClosed.connect(self.dock_closed_event_handler)
        self.psd_view = Dock(name='Power spectrum', closable=True)
        self.psd_view.sigClosed.connect(self.dock_closed_event_handler)
        self.stim_report_view = Dock(name='MEG analysis', closable=True)
        self.stim_report_view.sigClosed.connect(self.dock_closed_event_handler)
        self.meg_3d_view = Dock(name='3D viewer', closable=True)
        self.meg_3d_view.sigClosed.connect(self.dock_closed_event_handler)

        self.sig_view.addDock(self.chs_view, 'left')
        self.sig_view.addDock(self.log_view, 'below', self.chs_view)
        self.sig_view.addDock(self.stim_report_view, 'right')
        self.sig_view.addDock(self.psd_view, 'above', self.stim_report_view)
        self.sig_view.addDock(self.meg_3d_view, 'above', self.psd_view)

        self.sig_view_dock_status_dict = {}

        self.init_3d_layout()
        self.init_chs_layout()
        self.init_log_layout()
        self.init_power_spectrum_layout()
        self.init_stimulus_report_layout()

        self.sig_view_state = self.sig_view.saveState()

    def restore_ui(self):
        self.sig_view.restoreState(self.sig_view_state, missing='create')
        for dock, status in self.sig_view_dock_status_dict.items():
            if dock == 'Channels' and not status:
                self.sig_view.addDock(self.chs_view, 'left')
            elif dock == 'Log' and not status:
                self.sig_view.addDock(self.log_view, 'below', self.chs_view)
            elif dock in {'MEG analysis', 'Stimulus report'} and not status:
                self.sig_view.addDock(self.stim_report_view, 'right')
            elif dock == 'Power spectrum' and not status:
                self.sig_view.addDock(self.psd_view, 'above', self.stim_report_view)
            elif dock == '3D viewer' and not status:
                self.sig_view.addDock(self.meg_3d_view, 'above', self.psd_view)


    def dock_closed_event_handler(self, dock):
        self.sig_view_dock_status_dict[dock.name()] = False

    def clear_layout(self):
        """Clears the layout of the signal viewer."""
        for pw in self.pw_list:
            pw.clear()
        for vb in self.filtered_vb_list:
            vb.clear()
        self.vb_pics.clear()
        self.chs_layout.clear()

        self.chs_layout.ci.layout.setSpacing(0)
        self.chs_layout.ci.layout.setContentsMargins(0, 0, 0, 0)
            
    def init_chs_layout(self):
        """Initializes the signal viewer with a plot and separate curves for each channel."""
        self.filtering = Filtering(c=self.c)
        self.signal_viewer = MEGSignalViewer(
            c=self.c,
            chs_view=self.chs_view,
            meg_3d=self.meg_3d,
            filtering=self.filtering,
        )

    def init_log_layout(self):
        """Create the read-only application log beneath the Channels dock."""
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(5000)
        self.txt_log.setPlaceholderText('MEG processing messages will appear here.')
        self.log_view.addWidget(self.txt_log)
        

            


    def _reset_fif_analysis_state(self):
        """Reset cached MNE analysis products for currently loaded FIF."""
        clear_parcel_timecourse_cache(self)
        self._fif_events = np.empty((0, 3), dtype=np.int64)
        self._fif_event_id = {}
        self._fif_event_labels_by_code = {}
        self._fif_event_strategy = None
        self._fif_epochs = None
        self._fif_epochs_by_path = {}
        self._fif_epoch_metadata_by_path = {}
        self._fif_topomap_run_paths = []
        self._fif_evokeds = {}
        self._fif_primary_event = None
        self._fif_primary_evoked = None
        self._fif_psd = None
        self._fif_tfr = None
        self._fif_noise_cov = None
        self._fif_noise_cov_metadata = {}
        if not hasattr(self, "_fif_noise_cov_by_path"):
            self._fif_noise_cov_by_path = {}
        if not hasattr(self, "_fif_noise_cov_meta_by_path"):
            self._fif_noise_cov_meta_by_path = {}
        self._fif_analysis_summary = {}
        self._fif_filename = None
        self._fif_original_filename = None
        self._fif_ica_metadata = {}
        self._fif_trial_response_table = pd.DataFrame()
        self._fif_stimulus_summary_table = pd.DataFrame()

    def _reset_fif_analysis_outputs_in_3d(self):
        """Clear 3D FIF overlays/projections when switching data sources."""
        if hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, 'clear_meg_sensor_activity'):
            self.meg_3d.clear_meg_sensor_activity()
        if hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, 'clear_meg_source_projection'):
            self.meg_3d.clear_meg_source_projection()
        if hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, 'set_meg_noise_cov'):
            self.meg_3d.set_meg_noise_cov(None)
        if hasattr(self, 'power_spectrum') and self.power_spectrum is not None and hasattr(self.power_spectrum, 'clear_mne_psd_plot'):
            self.power_spectrum.clear_mne_psd_plot()
        if hasattr(self, 'meg_analysis') and self.meg_analysis is not None and hasattr(self.meg_analysis, 'clear'):
            self.meg_analysis.clear("No MEG analysis loaded.")
        elif hasattr(self, 'stimulus_report') and self.stimulus_report is not None and hasattr(self.stimulus_report, 'clear'):
            self.stimulus_report.clear("No stimulus-response report loaded.")

    def _detect_fif_events(self, raw):
        """Detect events from annotations/stim channels/fixed windows."""
        try:
            events, event_id = mne.events_from_annotations(raw, verbose='error')
            if events is not None and len(events) > 0:
                return np.asarray(events, dtype=np.int64), dict(event_id), 'annotations'
        except Exception:
            pass

        try:
            stim_picks = mne.pick_types(raw.info, stim=True, exclude=[])
            if stim_picks.size > 0:
                stim_names = [raw.ch_names[int(idx)] for idx in stim_picks]
                events = mne.find_events(
                    raw,
                    stim_channel=stim_names,
                    shortest_event=1,
                    min_duration=0.0,
                    verbose='error',
                )
                if events is not None and len(events) > 0:
                    unique_codes = sorted({int(code) for code in events[:, 2]})
                    event_id = {f"stim_{code}": int(code) for code in unique_codes}
                    return np.asarray(events, dtype=np.int64), event_id, 'stim'
        except Exception:
            pass

        duration = max(1.0, float(getattr(self.signal_viewer, 'length', 5.0)))
        try:
            events = mne.make_fixed_length_events(
                raw,
                id=1,
                duration=duration,
                overlap=0.0,
            )
            if events is not None and len(events) > 0:
                return np.asarray(events, dtype=np.int64), {'fixed': 1}, 'fixed_length'
        except Exception:
            pass

        return np.empty((0, 3), dtype=np.int64), {}, 'none'

    def _limit_fif_events_for_epoching(self, events, event_id, max_events=320, max_conditions=8):
        """Limit event volume/conditions to keep on-load analysis responsive."""
        if events is None or len(events) == 0:
            return np.empty((0, 3), dtype=np.int64), {}

        event_codes = events[:, 2].astype(int)
        unique_codes, counts = np.unique(event_codes, return_counts=True)
        ranked_codes = [
            int(code)
            for _, code in sorted(
                zip(counts, unique_codes),
                key=lambda item: (-int(item[0]), int(item[1])),
            )
        ]
        keep_codes = set(ranked_codes[:max_conditions])
        keep_mask = np.isin(event_codes, list(keep_codes))
        events_kept = np.asarray(events[keep_mask], dtype=np.int64)

        if len(events_kept) > max_events:
            keep_idx = np.linspace(0, len(events_kept) - 1, max_events, dtype=int)
            events_kept = events_kept[keep_idx]

        kept_unique = sorted({int(code) for code in events_kept[:, 2]})
        event_id_kept = {}
        for label, code in (event_id or {}).items():
            icode = int(code)
            if icode in kept_unique and label not in event_id_kept:
                event_id_kept[str(label)] = icode
        for code in kept_unique:
            if code not in event_id_kept.values():
                event_id_kept[f"event_{code}"] = int(code)
        return events_kept, event_id_kept

    def _log_fif_stim_spacing_check(self, events_abs, sfreq, context):
        """Print/log spacing diagnostics for stimulus events."""
        events = np.asarray(events_abs if events_abs is not None else [], dtype=np.int64)
        if events.ndim != 2 or events.shape[1] < 3 or events.shape[0] == 0:
            return

        sfreq = float(sfreq) if sfreq else 1.0
        tolerance_samples = 1.0
        codes = events[:, 2].astype(int)

        def emit(message):
            print(message)
            if hasattr(self, "c") and hasattr(self.c, "log"):
                self.c.log.emit(message)

        def format_values(values, max_values=8):
            values = np.asarray(values).ravel()
            if values.size == 0:
                return "[]"
            shown = ", ".join(str(int(v)) for v in values[:max_values])
            if values.size > max_values:
                shown += ", ..."
            return f"[{shown}]"

        def format_common_intervals(intervals, max_values=6):
            if intervals.size == 0:
                return "[]"
            unique, counts = np.unique(intervals.astype(np.int64), return_counts=True)
            ranked = sorted(
                zip(counts, unique),
                key=lambda item: (-int(item[0]), int(item[1])),
            )
            parts = [f"{int(interval)}x{int(count)}" for count, interval in ranked[:max_values]]
            if len(ranked) > max_values:
                parts.append("...")
            return "[" + ", ".join(parts) + "]"

        def event_samples(code):
            return np.sort(np.asarray(events[codes == int(code), 0], dtype=np.int64))

        def summarize_event(label, code):
            samples = event_samples(code)
            if samples.size == 0:
                emit(f"{context}: {label} code={int(code)} count=0 (not found).")
                return samples
            if samples.size == 1:
                emit(
                    f"{context}: {label} code={int(code)} count=1; "
                    "need at least 2 events to check spacing."
                )
                return samples

            intervals = np.diff(samples)
            median_interval = float(np.median(intervals))
            max_dev = float(np.max(np.abs(intervals.astype(float) - median_interval)))
            exact = bool(np.all(intervals == intervals[0]))
            within_tol = bool(max_dev <= tolerance_samples)

            message = (
                f"{context}: {label} code={int(code)} count={int(samples.size)}, "
                f"interval_samples median={median_interval:.3f}, "
                f"min={int(np.min(intervals))}, max={int(np.max(intervals))}, "
                f"max_dev_from_median={max_dev:.3f}, "
                f"interval_s median={median_interval / sfreq:.6f}, "
                f"min={float(np.min(intervals)) / sfreq:.6f}, "
                f"max={float(np.max(intervals)) / sfreq:.6f}, "
                f"exact_equispaced={'yes' if exact else 'no'}, "
                f"within_1_sample={'yes' if within_tol else 'no'}, "
                f"common_intervals={format_common_intervals(intervals)}, "
                f"first_intervals={format_values(intervals)}."
            )
            if not within_tol:
                off_idx = np.where(np.abs(intervals.astype(float) - median_interval) > tolerance_samples)[0]
                message += f" off_median_interval_indices={format_values(off_idx)}."
            emit(message)
            return samples

        stim_on_samples = summarize_event("stim_on", 2)
        stim_on_pix_samples = summarize_event("stim_on_pix", 4096)
        btn_p_samples = event_samples(512)
        emit(f"{context}: btn_p code=512 count={int(btn_p_samples.size)}.")

        n_pairs = min(int(stim_on_samples.size), int(stim_on_pix_samples.size))
        if n_pairs <= 0:
            emit(
                f"{context}: stim_on/stim_on_pix pairing unavailable; "
                f"stim_on_count={int(stim_on_samples.size)}, "
                f"stim_on_pix_count={int(stim_on_pix_samples.size)}."
            )
            return

        latencies = stim_on_pix_samples[:n_pairs] - stim_on_samples[:n_pairs]
        latency_median = float(np.median(latencies))
        latency_max_dev = float(np.max(np.abs(latencies.astype(float) - latency_median)))
        latency_exact = bool(np.all(latencies == latencies[0]))

        pair_message = (
            f"{context}: stim_on_pix - stim_on paired_by_order n={n_pairs}, "
            f"unmatched_stim_on={int(stim_on_samples.size) - n_pairs}, "
            f"unmatched_stim_on_pix={int(stim_on_pix_samples.size) - n_pairs}, "
            f"latency_samples median={latency_median:.3f}, "
            f"min={int(np.min(latencies))}, max={int(np.max(latencies))}, "
            f"max_dev_from_median={latency_max_dev:.3f}, "
            f"latency_s median={latency_median / sfreq:.6f}, "
            f"constant_latency={'yes' if latency_exact else 'no'}, "
            f"first_latencies={format_values(latencies)}."
        )

        n_interval_pairs = min(max(0, n_pairs - 1), int(stim_on_samples.size) - 1, int(stim_on_pix_samples.size) - 1)
        if n_interval_pairs > 0:
            stim_intervals = np.diff(stim_on_samples[: n_interval_pairs + 1])
            pix_intervals = np.diff(stim_on_pix_samples[: n_interval_pairs + 1])
            interval_delta = pix_intervals - stim_intervals
            pair_message += (
                f" pix_minus_stim_interval_delta_samples median={float(np.median(interval_delta)):.3f}, "
                f"min={int(np.min(interval_delta))}, max={int(np.max(interval_delta))}, "
                f"same_spacing={'yes' if bool(np.all(interval_delta == 0)) else 'no'}, "
                f"first_interval_deltas={format_values(interval_delta)}."
            )
        emit(pair_message)

    def _compute_topomap_epochs_for_fif(self, fif_filename=None, raw=None):
        """Compute stimulus-onset MEG epochs for one FIF for topomap/run selection."""
        fif_path = os.path.abspath(str(fif_filename)) if fif_filename else None
        if raw is None:
            if not fif_path:
                raise RuntimeError("FIF filename is required for topomap epoching.")
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="error")

        sfreq = float(raw.info["sfreq"])
        meg_picks, bad_meg_names = good_meg_picks(raw.info, meg=True, ref_meg=False)
        if bad_meg_names:
            emit_bad_meg_channel_log(
                getattr(getattr(self, "c", None), "log", None),
                "FIF topomap epochs",
                fif_path=fif_path,
                bad_names=bad_meg_names,
            )
        if meg_picks.size == 0:
            raise RuntimeError("No usable MEG channels are available for topomap epoching.")

        events_abs, _event_id, strategy = self._detect_fif_events(raw)
        if events_abs.size == 0:
            raise RuntimeError("No events detected.")

        first_samp = int(getattr(raw, "first_samp", 0))
        events_rel = np.asarray(events_abs, dtype=np.int64).copy()
        events_rel[:, 0] = np.asarray(events_rel[:, 0] - first_samp, dtype=np.int64)
        events_rel[:, 0] = np.clip(events_rel[:, 0], 0, max(0, int(raw.n_times) - 1))

        canonical_event_id = {
            "stim_on": 2,
            "stim_off": 4,
            "btn_p": 512,
            "stim_on_pix": 4096,
        }
        event_codes = events_abs[:, 2].astype(int)
        onset_code = int(canonical_event_id["stim_on"])
        offset_code = int(canonical_event_id["stim_off"])
        if not np.any(event_codes == onset_code):
            raise RuntimeError("No stim_on events (code 2) detected.")

        onset_samples = np.asarray(events_abs[event_codes == onset_code, 0], dtype=np.int64)
        offset_samples = np.asarray(events_abs[event_codes == offset_code, 0], dtype=np.int64)
        estimated_stim_duration = min(
            0.8,
            max(0.35, float(getattr(self.signal_viewer, "length", 5.0)) * 0.2),
        )
        if onset_samples.size > 0 and offset_samples.size > 0:
            next_offset_idx = np.searchsorted(offset_samples, onset_samples, side="right")
            next_onset = np.concatenate((onset_samples[1:], np.array([np.iinfo(np.int64).max], dtype=np.int64)))
            valid_match = next_offset_idx < offset_samples.size
            if np.any(valid_match):
                matched_offsets = offset_samples[next_offset_idx[valid_match]]
                matched_onsets = onset_samples[valid_match]
                matched_next_onsets = next_onset[valid_match]
                same_trial = (matched_offsets > matched_onsets) & (matched_offsets < matched_next_onsets)
                if np.any(same_trial):
                    durations_sec = (matched_offsets[same_trial] - matched_onsets[same_trial]) / sfreq
                    estimated_stim_duration = float(np.median(durations_sec))

        tmin = -0.1
        tmax = max(0.05, float(estimated_stim_duration))
        baseline = (-0.1, 0.0)
        if (tmax - tmin) >= (raw.n_times / sfreq):
            tmin = 0.0
            tmax = min(0.5, max(0.2, (raw.n_times / sfreq) * 0.5))
            baseline = None

        try:
            epochs = mne.Epochs(
                raw,
                events_abs,
                event_id={"stim_on": onset_code},
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=meg_picks,
                preload=True,
                reject_by_annotation=True,
                verbose="error",
            )
            if len(epochs) == 0:
                raise RuntimeError("No epochs survived initial epoching.")
        except Exception:
            tmin = 0.0
            tmax = min(0.5, max(0.2, float(getattr(self.signal_viewer, "length", 5.0)) * 0.1))
            baseline = None
            epochs = mne.Epochs(
                raw,
                events_abs,
                event_id={"stim_on": onset_code},
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=meg_picks,
                preload=True,
                reject_by_annotation=True,
                verbose="error",
            )
        if epochs is None or len(epochs) == 0:
            raise RuntimeError("Epoching produced no usable epochs.")

        meta = {
            "path": fif_path,
            "run_id": self._infer_run_id_from_fif_filename(fif_path) if fif_path else None,
            "event_strategy": str(strategy),
            "n_events": int(len(events_abs)),
            "n_epochs": int(len(epochs)),
            "tmin": float(tmin),
            "tmax": float(tmax),
            "baseline": tuple(float(v) for v in baseline) if baseline is not None else None,
            "estimated_stim_duration": float(estimated_stim_duration),
            "event_id": dict(canonical_event_id),
        }
        return epochs, meta

    def _prepare_topomap_epoch_runs(self, fif_files):
        """Cache topomap epochs for the active FIF plus any included report FIFs."""
        ordered = []
        for path in fif_files or []:
            if not path:
                continue
            path_abs = os.path.abspath(str(path))
            if path_abs not in ordered:
                ordered.append(path_abs)

        if not hasattr(self, "_fif_epochs_by_path") or not isinstance(self._fif_epochs_by_path, dict):
            self._fif_epochs_by_path = {}
        if not hasattr(self, "_fif_epoch_metadata_by_path") or not isinstance(self._fif_epoch_metadata_by_path, dict):
            self._fif_epoch_metadata_by_path = {}

        current_path = os.path.abspath(str(getattr(self, "_fif_filename", "") or "")) if getattr(self, "_fif_filename", None) else None
        if current_path and getattr(self, "_fif_epochs", None) is not None:
            self._fif_epochs_by_path[current_path] = self._fif_epochs
            self._fif_epoch_metadata_by_path.setdefault(
                current_path,
                {
                    "path": current_path,
                    "run_id": self._infer_run_id_from_fif_filename(current_path),
                    "n_epochs": int(len(self._fif_epochs)),
                    "event_strategy": str(getattr(self, "_fif_event_strategy", "") or ""),
                },
            )

        for path_abs in ordered:
            if path_abs in self._fif_epochs_by_path:
                continue
            try:
                epochs, meta = self._compute_topomap_epochs_for_fif(path_abs)
            except Exception as exc:
                self.c.log.emit(
                    f"FIF topomap epochs unavailable for {os.path.basename(path_abs)}: {exc}"
                )
                continue
            self._fif_epochs_by_path[path_abs] = epochs
            self._fif_epoch_metadata_by_path[path_abs] = dict(meta or {})

        self._fif_topomap_run_paths = [
            path for path in ordered if path in self._fif_epochs_by_path
        ]
        for path in self._fif_epochs_by_path:
            if path not in self._fif_topomap_run_paths:
                self._fif_topomap_run_paths.append(path)

    def _noise_cov_cache_entries(self):
        if not hasattr(self, "_fif_noise_cov_by_path") or not isinstance(self._fif_noise_cov_by_path, dict):
            self._fif_noise_cov_by_path = {}
        if not hasattr(self, "_fif_noise_cov_meta_by_path") or not isinstance(self._fif_noise_cov_meta_by_path, dict):
            self._fif_noise_cov_meta_by_path = {}
        return self._fif_noise_cov_by_path, self._fif_noise_cov_meta_by_path

    @staticmethod
    def _selected_covariance_method(noise_cov):
        try:
            method = noise_cov.get("method", "")
        except Exception:
            method = ""
        if isinstance(method, (list, tuple)):
            method = ",".join(str(v) for v in method)
        return str(method or "").strip()

    def _canonical_covariance_events(self, raw, events_abs=None):
        """Return stimulus-onset events to use for baseline noise covariance."""
        if events_abs is None:
            events_abs, _event_id, _strategy = self._detect_fif_events(raw)
        events = np.asarray(events_abs if events_abs is not None else [], dtype=np.int64)
        if events.ndim != 2 or events.shape[1] < 3 or events.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64), None

        codes = events[:, 2].astype(int)
        if np.any(codes == 2):
            cov_code = 2
            cov_label = "stim_on"
        else:
            unique_codes, counts = np.unique(codes, return_counts=True)
            cov_code = int(unique_codes[int(np.argmax(counts))])
            cov_label = f"event_{cov_code}"
        return np.asarray(events[codes == cov_code], dtype=np.int64), {cov_label: int(cov_code)}

    def _compute_tutorial_noise_covariance_for_fif(
        self,
        fif_filename=None,
        raw=None,
        events_abs=None,
        picks=None,
    ):
        """Compute run-specific noise covariance using the MNE tutorial baseline pattern."""
        if raw is None:
            if not fif_filename:
                raise RuntimeError("FIF filename is required for noise covariance.")
            raw = mne.io.read_raw_fif(fif_filename, preload=False, verbose="error")
        info = raw.info
        meg_picks = np.asarray(picks if picks is not None else [], dtype=int)
        if meg_picks.size == 0:
            meg_picks, bad_meg_names = good_meg_picks(info, meg=True, ref_meg=False)
            if bad_meg_names:
                emit_bad_meg_channel_log(
                    getattr(getattr(self, "c", None), "log", None),
                    "Noise covariance",
                    fif_path=fif_filename,
                    bad_names=bad_meg_names,
                )
        if meg_picks.size == 0:
            raise RuntimeError("No usable MEG channels are available for noise covariance.")
        cov_channel_names = [str(info["ch_names"][int(idx)]) for idx in meg_picks]

        events_cov, event_id_cov = self._canonical_covariance_events(raw, events_abs=events_abs)
        baseline_errors = []
        if events_cov.size > 0 and event_id_cov:
            for baseline_tmin in (-0.2, -0.1, -0.05):
                try:
                    epochs_cov = mne.Epochs(
                        raw,
                        events_cov,
                        event_id=event_id_cov,
                        tmin=float(baseline_tmin),
                        tmax=0.0,
                        baseline=None,
                        picks=meg_picks,
                        preload=True,
                        reject_by_annotation=True,
                        event_repeated="drop",
                        verbose="error",
                    )
                    if len(epochs_cov) == 0:
                        raise RuntimeError("no baseline epochs survived")
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            message="divide by zero encountered.*",
                        )
                        noise_cov = mne.compute_covariance(
                            epochs_cov,
                            tmax=0.0,
                            method=["shrunk", "empirical"],
                            rank=None,
                            verbose="error",
                        )
                    meta = {
                        "kind": "baseline_epochs",
                        "method": "baseline_epochs_shrunk_empirical_rank_none",
                        "selected_method": self._selected_covariance_method(noise_cov),
                        "rank": None,
                        "baseline_tmin": float(baseline_tmin),
                        "baseline_tmax": 0.0,
                        "n_epochs": int(len(epochs_cov)),
                        "event_id": {str(k): int(v) for k, v in event_id_cov.items()},
                        "n_channels": int(len(getattr(noise_cov, "ch_names", []) or [])),
                        "channel_names": cov_channel_names,
                    }
                    return noise_cov, meta
                except Exception as exc:
                    baseline_errors.append(f"{baseline_tmin:.3f}s: {exc}")

        cov_tmax = min(float(raw.times[-1]), 60.0)
        noise_cov = mne.compute_raw_covariance(
            raw,
            picks=meg_picks,
            tmin=0.0,
            tmax=cov_tmax,
            method="empirical",
            rank=None,
            verbose="error",
        )
        meta = {
            "kind": "raw_fallback",
            "method": "raw_empirical_rank_none",
            "selected_method": self._selected_covariance_method(noise_cov),
            "rank": None,
            "raw_tmin": 0.0,
            "raw_tmax": float(cov_tmax),
            "n_epochs": 0,
            "event_id": {},
            "n_channels": int(len(getattr(noise_cov, "ch_names", []) or [])),
            "channel_names": cov_channel_names,
            "fallback_reason": "; ".join(baseline_errors) if baseline_errors else "no covariance events available",
        }
        return noise_cov, meta

    def _get_or_compute_noise_cov_for_fif(
        self,
        fif_filename=None,
        raw=None,
        events_abs=None,
        picks=None,
        force=False,
    ):
        """Return a run-specific noise covariance and metadata for one FIF."""
        if fif_filename:
            path_abs = os.path.abspath(str(fif_filename))
        elif raw is not None and getattr(raw, "filenames", None):
            path_abs = os.path.abspath(str(raw.filenames[0]))
        else:
            path_abs = os.path.abspath(str(getattr(self, "_fif_filename", "") or ""))
        if not path_abs:
            raise RuntimeError("FIF filename is unavailable for noise covariance.")

        cov_cache, meta_cache = self._noise_cov_cache_entries()
        fif_identity = self._file_identity_for_metadata(path_abs)
        cached_meta = meta_cache.get(path_abs)
        cached_cov = cov_cache.get(path_abs)
        if (
            not force
            and cached_cov is not None
            and isinstance(cached_meta, dict)
            and cached_meta.get("fif_identity") == fif_identity
        ):
            try:
                return cached_cov.copy(), dict(cached_meta)
            except Exception:
                return cached_cov, dict(cached_meta)

        noise_cov, meta = self._compute_tutorial_noise_covariance_for_fif(
            fif_filename=path_abs,
            raw=raw,
            events_abs=events_abs,
            picks=picks,
        )
        meta = dict(meta or {})
        meta["fif_path"] = path_abs
        meta["fif_identity"] = fif_identity
        meta["created_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            cov_cache[path_abs] = noise_cov.copy()
        except Exception:
            cov_cache[path_abs] = noise_cov
        meta_cache[path_abs] = dict(meta)
        if getattr(self, "_fif_filename", None) and os.path.abspath(str(self._fif_filename)) == path_abs:
            try:
                self._fif_noise_cov = noise_cov.copy()
            except Exception:
                self._fif_noise_cov = noise_cov
            self._fif_noise_cov_metadata = dict(meta)
        return noise_cov, meta

    def _cached_noise_cov_metadata_for_fif(self, fif_filename):
        if not fif_filename:
            return {}
        _cov_cache, meta_cache = self._noise_cov_cache_entries()
        path_abs = os.path.abspath(str(fif_filename))
        meta = meta_cache.get(path_abs)
        return dict(meta) if isinstance(meta, dict) else {}

    def _compute_fif_mne_pipeline(self):
        """Compute MNE analysis products for loaded FIF data."""
        if not getattr(self, "b_fif_loaded", False) or not hasattr(self, "_fif_raw"):
            return False

        raw = self._fif_raw
        sfreq = float(raw.info["sfreq"])
        meg_picks, bad_meg_names = good_meg_picks(raw.info, meg=True, ref_meg=False)
        if bad_meg_names:
            emit_bad_meg_channel_log(
                self.c.log,
                "FIF MNE pipeline",
                fif_path=getattr(self, "_fif_filename", None),
                bad_names=bad_meg_names,
            )
        if meg_picks.size == 0:
            self.c.log.emit("FIF MNE pipeline skipped: no MEG channels available.")
            return False

        events_abs, event_id, strategy = self._detect_fif_events(raw)
        if events_abs.size == 0:
            self.c.log.emit("FIF MNE pipeline: no events detected.")
            return False

        # events_abs, event_id = self._limit_fif_events_for_epoching(
        #     events=events_abs,
        #     event_id=event_id,
        #     max_events=320,
        #     max_conditions=8,
        # )
        # if events_abs.size == 0 or not event_id:
        #     self.c.log.emit("FIF MNE pipeline: events were filtered out after limits.")
        #     return False

        first_samp = int(getattr(raw, "first_samp", 0))
        events_rel = np.asarray(events_abs, dtype=np.int64).copy()
        events_rel[:, 0] = np.asarray(events_rel[:, 0] - first_samp, dtype=np.int64)
        events_rel[:, 0] = np.clip(events_rel[:, 0], 0, max(0, int(raw.n_times) - 1))

        # Canonical event-code mapping for task semantics.
        # 2: stimulus onset, 4: stimulus offset, 512: button press, 4096: pixel onset.
        canonical_event_id = {
            "stim_on": 2,
            "stim_off": 4,
            "btn_p": 512,
            "stim_on_pix": 4096,
        }
        code_to_label = {}
        for label, code in canonical_event_id.items():
            code_to_label[int(code)] = str(label)

        self._fif_events = events_rel
        self._fif_event_id = dict(canonical_event_id)
        self._fif_event_labels_by_code = dict(code_to_label)
        self._fif_event_strategy = str(strategy)

        event_codes = events_abs[:, 2].astype(int)
        self._log_fif_stim_spacing_check(
            events_abs=events_abs,
            sfreq=sfreq,
            context="FIF MNE pipeline event spacing check",
        )
        onset_code = int(canonical_event_id["stim_on"])
        offset_code = int(canonical_event_id["stim_off"])
        if not np.any(event_codes == onset_code):
            self.c.log.emit("FIF MNE pipeline: no stim_on events (code 2) detected.")
            return False

        epoch_event_id = {"stim_on": onset_code}
        onset_samples = np.asarray(events_abs[event_codes == onset_code, 0], dtype=np.int64)
        offset_samples = np.asarray(events_abs[event_codes == offset_code, 0], dtype=np.int64)

        estimated_stim_duration = min(
            0.8,
            max(0.35, float(getattr(self.signal_viewer, "length", 5.0)) * 0.2),
        )
        if onset_samples.size > 0 and offset_samples.size > 0:
            next_offset_idx = np.searchsorted(offset_samples, onset_samples, side="right")
            next_onset = np.concatenate((onset_samples[1:], np.array([np.iinfo(np.int64).max], dtype=np.int64)))
            valid_match = next_offset_idx < offset_samples.size
            if np.any(valid_match):
                matched_offsets = offset_samples[next_offset_idx[valid_match]]
                matched_onsets = onset_samples[valid_match]
                matched_next_onsets = next_onset[valid_match]
                same_trial = (matched_offsets > matched_onsets) & (matched_offsets < matched_next_onsets)
                if np.any(same_trial):
                    durations_sec = (matched_offsets[same_trial] - matched_onsets[same_trial]) / sfreq
                    estimated_stim_duration = float(np.median(durations_sec))
                    self.c.log.emit(
                        f"FIF MNE pipeline: estimated stimulus duration from onset->offset is "
                        f"{estimated_stim_duration:.3f}s (median, n={int(np.sum(same_trial))})."
                    )

        tmin = -0.1
        tmax = max(0.05, float(estimated_stim_duration))
        baseline = (-0.1, 0.0)
        if (tmax - tmin) >= (raw.n_times / sfreq):
            tmin = 0.0
            tmax = min(0.5, max(0.2, (raw.n_times / sfreq) * 0.5))
            baseline = None

        epochs = None
        try:
            epochs = mne.Epochs(
                raw,
                events_abs,
                event_id=epoch_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=meg_picks,
                preload=True,
                reject_by_annotation=True,
                verbose='error',
            )
            if len(epochs) == 0:
                raise RuntimeError("No epochs survived initial epoching.")
        except Exception:
            tmin = 0.0
            tmax = min(0.5, max(0.2, float(getattr(self.signal_viewer, "length", 5.0)) * 0.1))
            baseline = None
            epochs = mne.Epochs(
                raw,
                events_abs,
                event_id=epoch_event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                picks=meg_picks,
                preload=True,
                reject_by_annotation=True,
                verbose='error',
            )

        if epochs is None or len(epochs) == 0:
            self.c.log.emit("FIF MNE pipeline: epoching produced no usable epochs.")
            return False

        evokeds = {}
        event_counts = {}
        for label, code in epochs.event_id.items():
            label_str = str(label)
            cond_indices = np.where(epochs.events[:, 2] == int(code))[0]
            n_epochs = int(len(cond_indices))
            if n_epochs <= 0:
                continue
            event_counts[label_str] = n_epochs
            try:
                evokeds[label_str] = epochs[cond_indices].average()
            except Exception:
                continue

        if not evokeds:
            evokeds["all"] = epochs.average()
            event_counts["all"] = int(len(epochs))

        primary_event = max(event_counts.items(), key=lambda item: item[1])[0]
        primary_evoked = evokeds[primary_event]

        psd = None
        try:
            psd_picks = meg_picks
            if psd_picks.size > 0:
                raw_psd = raw.copy().pick(psd_picks)
                psd_tmax = min(float(raw_psd.times[-1]), 60.0)
                psd = raw_psd.compute_psd(
                    fmin=1.0,
                    fmax=120.0,
                    tmin=0.0,
                    tmax=psd_tmax,
                    verbose='error',
                )
        except Exception:
            psd = None

        tfr_power = None
        try:
            if primary_event in epochs.event_id:
                primary_code = int(epochs.event_id[primary_event])
                primary_idx = np.where(epochs.events[:, 2] == primary_code)[0]
                tfr_epochs = epochs[primary_idx] if len(primary_idx) > 0 else epochs
            else:
                tfr_epochs = epochs
            if len(tfr_epochs.ch_names) > 0:
                n_tfr_channels = min(24, len(tfr_epochs.ch_names))
                tfr_epochs_small = tfr_epochs.copy().pick(tfr_epochs.ch_names[:n_tfr_channels])
                freqs = np.geomspace(4.0, 80.0, num=18)
                n_cycles = np.maximum(freqs / 2.0, 2.0)
                decim = max(1, int(round(float(tfr_epochs_small.info["sfreq"]) / 200.0)))
                tfr_power = tfr_epochs_small.compute_tfr(
                    method="morlet",
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    output="power",
                    average=True,
                    return_itc=False,
                    decim=decim,
                    verbose='error',
                )
        except Exception:
            tfr_power = None

        noise_cov = None
        noise_cov_meta = {}
        try:
            noise_cov, noise_cov_meta = self._get_or_compute_noise_cov_for_fif(
                fif_filename=getattr(self, "_fif_filename", None),
                raw=raw,
                events_abs=events_abs,
                picks=meg_picks,
                force=True,
            )
            if isinstance(noise_cov_meta, dict) and noise_cov_meta.get("kind") == "raw_fallback":
                self.c.log.emit(
                    "FIF noise covariance warning: using raw covariance fallback; "
                    f"{noise_cov_meta.get('fallback_reason', 'baseline covariance unavailable')}"
                )
        except Exception as exc:
            noise_cov = None
            noise_cov_meta = {"kind": "unavailable", "method": "", "error": str(exc)}
            self.c.log.emit(f"FIF noise covariance unavailable: {exc}")

        self._fif_epochs = epochs
        self._fif_evokeds = evokeds
        self._fif_primary_event = primary_event
        self._fif_primary_evoked = primary_evoked
        self._fif_psd = psd
        self._fif_tfr = tfr_power
        self._fif_noise_cov = noise_cov
        self._fif_noise_cov_metadata = dict(noise_cov_meta or {})
        self._fif_analysis_summary = {
            "event_strategy": strategy,
            "n_events": int(len(events_abs)),
            "n_epochs": int(len(epochs)),
            "n_conditions": int(len(evokeds)),
            "primary_event": str(primary_event),
            "n_psd_channels": int(len(psd.ch_names)) if psd is not None else 0,
            "n_tfr_channels": int(tfr_power.data.shape[0]) if tfr_power is not None else 0,
            "n_tfr_freqs": int(tfr_power.data.shape[1]) if tfr_power is not None else 0,
            "has_noise_cov": bool(noise_cov is not None),
            "noise_cov_kind": str(noise_cov_meta.get("kind", "")) if isinstance(noise_cov_meta, dict) else "",
            "noise_cov_method": str(noise_cov_meta.get("method", "")) if isinstance(noise_cov_meta, dict) else "",
            "noise_cov_selected_method": (
                str(noise_cov_meta.get("selected_method", "")) if isinstance(noise_cov_meta, dict) else ""
            ),
            "noise_cov_epochs": int(noise_cov_meta.get("n_epochs", 0)) if isinstance(noise_cov_meta, dict) else 0,
        }

        projected = False
        if hasattr(self, 'meg_3d') and self.meg_3d is not None:
            try:
                if hasattr(self.meg_3d, 'set_meg_noise_cov'):
                    self.meg_3d.set_meg_noise_cov(noise_cov, metadata=noise_cov_meta)
            except Exception as exc:
                self.c.log.emit(f"FIF source projection warning: {exc}")

        self._fif_analysis_summary["source_projected"] = bool(projected)
        return True

    def _fif_window_events(self, start_sample, stop_sample):
        """Return event payload in current FIF display window."""
        if self._fif_events is None or len(self._fif_events) == 0:
            return {'data': [], 'timestamp': []}

        start_sample = int(start_sample)
        stop_sample = int(stop_sample)
        if stop_sample <= start_sample:
            return {'data': [], 'timestamp': []}

        mask = (self._fif_events[:, 0] >= start_sample) & (self._fif_events[:, 0] < stop_sample)
        if not np.any(mask):
            return {'data': [], 'timestamp': []}

        window_events = self._fif_events[mask]
        event_data = []
        event_timestamps = []
        for sample, _, code in window_events:
            code_int = int(code)
            event_label = self._fif_event_labels_by_code.get(code_int, code_int)
            event_data.append(event_label)
            event_timestamps.append(int(sample))
        return {'data': event_data, 'timestamp': event_timestamps}









    def _compute_stimulus_report_from_event_logs(self, fif_filename, report_fif_files=None):
        """Build stimulus report from one or many FIF files in block order."""
        if not getattr(self, "b_fif_loaded", False):
            return False
        if not hasattr(self, "stimulus_report") or self.stimulus_report is None:
            return False

        if report_fif_files is None:
            report_fif_files = []
        ordered_paths = [os.path.abspath(str(fif_filename))]
        for path in report_fif_files:
            if not path:
                continue
            path_abs = os.path.abspath(str(path))
            if path_abs not in ordered_paths:
                ordered_paths.append(path_abs)

        trial_frames = []
        per_file_meta = []
        skipped = []
        for path in ordered_paths:
            try:
                trial_df, meta, err_msg = self._compute_trial_response_table_for_fif(path, max_report_channels=None)
            except Exception as exc:
                trial_df, meta, err_msg = pd.DataFrame(), {"source_fif_path": path}, str(exc)
            if trial_df.empty:
                skipped.append((path, err_msg or "no mapped trials"))
                continue
            trial_frames.append(trial_df)
            per_file_meta.append(meta)

        if not trial_frames:
            reason = "No mapped trials from events_log + FIF triggers."
            if skipped:
                first_reason = skipped[0][1]
                if first_reason:
                    reason = f"{reason} ({first_reason})"
            self.stimulus_report.clear(reason)
            return False

        trial_df_all = pd.concat(trial_frames, ignore_index=True, sort=False)
        if "trial_index" in trial_df_all.columns:
            trial_df_all["trial_index_in_file"] = pd.to_numeric(
                trial_df_all["trial_index"], errors="coerce"
            ).fillna(0).astype(int)
        trial_df_all["trial_index"] = np.arange(1, len(trial_df_all) + 1, dtype=int)

        primary_meta = per_file_meta[0] if per_file_meta else {}
        summary_meta = dict(primary_meta)
        summary_meta["n_fif_requested"] = int(len(ordered_paths))
        summary_meta["n_fif_processed"] = int(len(trial_frames))
        summary_meta["mapped_trials"] = int(len(trial_df_all))
        summary_meta["csv_onsets"] = int(sum(int(m.get("csv_onsets", 0)) for m in per_file_meta))
        summary_meta["fif_onsets"] = int(sum(int(m.get("fif_onsets", 0)) for m in per_file_meta))
        if len(trial_frames) > 1:
            summary_meta["run_id"] = "multi"
            summary_meta["block_id"] = "multi"

        self._fif_trial_response_table = trial_df_all.copy()
        self.stimulus_report.set_trial_data(trial_df_all, metadata=summary_meta)
        self._fif_stimulus_summary_table = self.stimulus_report.get_summary_df()

        if skipped:
            skip_msg = ", ".join([f"{os.path.basename(p)} ({msg})" for p, msg in skipped[:4]])
            self.c.log.emit(f"Stimulus report skipped {len(skipped)} FIF file(s): {skip_msg}")
        return True

    def _find_anatomy_subject_for_fif(self, fif_filename, require_bem=True):
        """Find a FreeSurfer subject under `../Anatomy` for a FIF path."""
        if not fif_filename:
            return None, None, None
        fif_dir = os.path.dirname(os.path.abspath(str(fif_filename)))
        parent_dir = os.path.dirname(fif_dir)
        anatomy_dir = os.path.join(parent_dir, "Anatomy")
        if not os.path.isdir(anatomy_dir):
            return None, None, anatomy_dir

        # Support flat layout where Anatomy itself is a subject root:
        #   Anatomy/surf, Anatomy/mri, Anatomy/bem
        if (
            os.path.isdir(os.path.join(anatomy_dir, "surf"))
            and os.path.isdir(os.path.join(anatomy_dir, "mri"))
            and (not require_bem or os.path.isdir(os.path.join(anatomy_dir, "bem")))
        ):
            return parent_dir, os.path.basename(anatomy_dir), anatomy_dir

        candidates = [
            anatomy_dir,
            os.path.join(anatomy_dir, "freesurfer", "subjects"),
            os.path.join(anatomy_dir, "FreeSurfer", "subjects"),
            os.path.join(anatomy_dir, "freesurfer"),
            os.path.join(anatomy_dir, "FreeSurfer"),
            os.path.join(anatomy_dir, "subjects"),
        ]
        uniq_candidates = []
        for cand in candidates:
            if cand not in uniq_candidates:
                uniq_candidates.append(cand)

        run_hint = os.path.basename(os.path.dirname(os.path.abspath(str(fif_filename)))).strip().lower()
        for subjects_dir in uniq_candidates:
            if not os.path.isdir(subjects_dir):
                continue
            subject_ids = []
            for subject_id in sorted(os.listdir(subjects_dir)):
                if subject_id.startswith("."):
                    continue
                subject_ids.append(subject_id)

            def _subject_rank(subj):
                low = str(subj).lower()
                if run_hint and low == run_hint:
                    return (0, low)
                if run_hint and run_hint in low:
                    return (1, low)
                return (2, low)

            # Prefer patient subjects before fsaverage; use fsaverage only as fallback.
            non_fsaverage = [s for s in subject_ids if str(s).lower() != "fsaverage"]
            fsaverage_subjects = [s for s in subject_ids if str(s).lower() == "fsaverage"]
            ordered_subjects = sorted(non_fsaverage, key=_subject_rank) + fsaverage_subjects

            for subject_id in ordered_subjects:
                subj_path = os.path.join(subjects_dir, subject_id)
                if not os.path.isdir(subj_path):
                    continue
                if (
                    os.path.isdir(os.path.join(subj_path, "surf"))
                    and os.path.isdir(os.path.join(subj_path, "mri"))
                    and (not require_bem or os.path.isdir(os.path.join(subj_path, "bem")))
                ):
                    return subjects_dir, subject_id, anatomy_dir

        return None, None, anatomy_dir


    @staticmethod
    def _inverse_operator_metadata_path(inverse_path):
        if not inverse_path:
            return None
        return f"{os.path.abspath(str(inverse_path))}.source_meta.json"

    @staticmethod
    def _serialize_transform_signature(signature):
        if signature is None:
            return None
        if isinstance(signature, np.ndarray):
            signature = signature.reshape(-1).tolist()
        elif isinstance(signature, tuple):
            signature = list(signature)
        elif not isinstance(signature, list):
            try:
                return float(signature)
            except Exception:
                return str(signature)

        try:
            return [float(v) for v in signature]
        except Exception:
            return [str(v) for v in signature]

    @staticmethod
    def _transform_signatures_match(current_signature, saved_signature):
        if current_signature is None or saved_signature is None:
            return current_signature is None and saved_signature is None
        try:
            current_arr = np.asarray(current_signature, dtype=float)
            saved_arr = np.asarray(saved_signature, dtype=float)
            return (
                current_arr.shape == saved_arr.shape
                and np.allclose(current_arr, saved_arr, rtol=0.0, atol=1e-12)
            )
        except Exception:
            def _as_text_list(value):
                if isinstance(value, np.ndarray):
                    value = value.reshape(-1).tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                elif not isinstance(value, list):
                    value = [value]
                return [str(v) for v in value]

            return _as_text_list(current_signature) == _as_text_list(saved_signature)

    def _current_meg_transform_metadata_for_fif(self, fif_filename):
        """Return the transform identity/signature currently selected for a FIF."""
        if not fif_filename or not hasattr(self, "meg_3d") or self.meg_3d is None:
            return None, None, None
        meg_3d = self.meg_3d
        target_identity = None
        if hasattr(meg_3d, "_fif_file_identity"):
            try:
                target_identity = meg_3d._fif_file_identity(fif_filename)
            except Exception:
                target_identity = None

        trans = None
        trans_path = None
        current_identity = getattr(meg_3d, "_meg_current_fif_identity", None)
        trans_identity = getattr(meg_3d, "_meg_trans_fif_identity", None)
        if (
            target_identity is not None
            and current_identity == target_identity
            and trans_identity == target_identity
        ):
            trans = getattr(meg_3d, "_meg_trans", None)
            trans_path = getattr(meg_3d, "_meg_loaded_trans_path", None)
            if trans is not None and trans_path and hasattr(meg_3d, "_fif_file_identity"):
                try:
                    current_trans_identity = meg_3d._fif_file_identity(trans_path)
                    loaded_trans_identity = getattr(meg_3d, "_meg_loaded_trans_file_identity", None)
                    if loaded_trans_identity is not None and current_trans_identity != loaded_trans_identity:
                        if hasattr(meg_3d, "_load_meg_trans_file"):
                            trans = meg_3d._load_meg_trans_file(trans_path, fif_path=fif_filename)
                except Exception:
                    pass

        if trans is None and hasattr(meg_3d, "_find_existing_meg_trans_file"):
            subject_id = None
            try:
                subject_id, _subjects_dir = meg_3d._get_brain_subject_info()
            except Exception:
                subject_id = None
            try:
                trans_path = meg_3d._find_existing_meg_trans_file(
                    fif_path=fif_filename,
                    subject_id=subject_id,
                )
            except Exception:
                trans_path = None
            if trans_path:
                try:
                    trans = mne.read_trans(str(trans_path))
                except Exception:
                    trans = None

        trans_file_identity = self._file_identity_for_metadata(trans_path) if trans_path else None
        trans_signature = None
        if trans is not None and hasattr(meg_3d, "_transform_signature"):
            try:
                trans_signature = meg_3d._transform_signature(trans)
            except Exception:
                trans_signature = None
        return trans_file_identity, trans_signature, os.path.abspath(str(trans_path)) if trans_path else None

    def _reload_meg_transform_for_fif(self, fif_filename):
        """Reload the matching run-specific MEG transform from disk if available."""
        if not fif_filename or not hasattr(self, "meg_3d") or self.meg_3d is None:
            return None
        meg_3d = self.meg_3d
        if not hasattr(meg_3d, "_find_existing_meg_trans_file") or not hasattr(meg_3d, "_load_meg_trans_file"):
            return None
        subject_id = None
        try:
            subject_id, _subjects_dir = meg_3d._get_brain_subject_info()
        except Exception:
            subject_id = None
        try:
            trans_path = meg_3d._find_existing_meg_trans_file(
                fif_path=fif_filename,
                subject_id=subject_id,
            )
        except Exception:
            trans_path = None
        if not trans_path:
            return None
        try:
            trans = meg_3d._load_meg_trans_file(trans_path, fif_path=fif_filename)
            meg_3d._meg_sensor_trans = trans
            meg_3d._meg_inverse_key = None
            meg_3d._meg_inverse_operator = None
            return os.path.abspath(str(trans_path))
        except Exception:
            return None

    def _inverse_operator_matches_fif(self, inverse_path, fif_filename):
        meta_path = self._inverse_operator_metadata_path(inverse_path)
        if not meta_path or not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False

        current_fif = self._file_identity_for_metadata(fif_filename)
        saved_fif = meta.get("fif_identity")
        if not isinstance(saved_fif, dict) or saved_fif != current_fif:
            return False
        try:
            version = int(meta.get("version", 0))
        except Exception:
            version = 0
        if version < 3:
            return False

        saved_noise_cov = meta.get("noise_covariance")
        if not isinstance(saved_noise_cov, dict):
            return False
        if saved_noise_cov.get("fif_identity") != current_fif:
            return False
        if str(saved_noise_cov.get("method", "")).strip() not in {
            "baseline_epochs_shrunk_empirical_rank_none",
            "raw_empirical_rank_none",
        }:
            return False

        saved_trans = meta.get("trans_identity")
        saved_signature = meta.get("trans_signature")
        current_trans, current_signature, _current_trans_path = self._current_meg_transform_metadata_for_fif(
            fif_filename
        )
        if current_trans is not None:
            if not isinstance(saved_trans, dict) or saved_trans != current_trans:
                return False
        elif isinstance(saved_trans, dict) and saved_trans.get("path"):
            current_trans = self._file_identity_for_metadata(saved_trans.get("path"))
            if current_trans != saved_trans:
                return False

        if current_signature is not None and saved_signature is not None:
            if not self._transform_signatures_match(current_signature, saved_signature):
                return False
        elif current_signature is not None:
            return False

        saved_channels = meta.get("channel_names")
        if not isinstance(saved_channels, list) or not saved_channels:
            return False
        saved_cov_channels = saved_noise_cov.get("channel_names")
        if isinstance(saved_cov_channels, list) and saved_cov_channels:
            if [str(ch) for ch in saved_cov_channels] != [str(ch) for ch in saved_channels]:
                return False
        try:
            use_loaded_raw = (
                getattr(self, "_fif_raw", None) is not None
                and os.path.abspath(str(getattr(self, "_fif_filename", "") or "")) == os.path.abspath(str(fif_filename))
            )
            if use_loaded_raw:
                info = self._fif_raw.info
            else:
                raw = mne.io.read_raw_fif(fif_filename, preload=False, verbose="error")
                info = raw.info
            current_channels, _bad_names = good_meg_channel_names(info, meg=True, ref_meg=False)
        except Exception:
            current_channels = []
        if [str(ch) for ch in saved_channels] != [str(ch) for ch in current_channels]:
            return False
        return True

    def _write_inverse_operator_metadata(self, inverse_path, fif_filename, channel_names=None):
        meta_path = self._inverse_operator_metadata_path(inverse_path)
        if not meta_path:
            return
        trans_identity, trans_signature, trans_path = self._current_meg_transform_metadata_for_fif(
            fif_filename
        )
        subject = None
        subjects_dir = None
        try:
            subject, subjects_dir = self.meg_3d._get_brain_subject_info()
        except Exception:
            pass
        excluded_bad_meg_channels = []
        try:
            raw = mne.io.read_raw_fif(fif_filename, preload=False, verbose="error")
            _good_names, excluded_bad_meg_channels = good_meg_channel_names(
                raw.info,
                meg=True,
                ref_meg=False,
            )
        except Exception:
            excluded_bad_meg_channels = []
        noise_cov_meta = self._cached_noise_cov_metadata_for_fif(fif_filename)
        if not noise_cov_meta:
            noise_cov_meta = {
                "method": "",
                "kind": "unavailable",
                "fif_identity": self._file_identity_for_metadata(fif_filename),
            }
        meta = {
            "kind": "mas_meg_inverse_operator",
            "version": 3,
            "fif_identity": self._file_identity_for_metadata(fif_filename),
            "trans_identity": trans_identity,
            "trans_signature": self._serialize_transform_signature(trans_signature),
            "subject": str(subject) if subject is not None else "",
            "subjects_dir": os.path.abspath(str(subjects_dir)) if subjects_dir else "",
            "channel_names": [str(ch) for ch in channel_names or []],
            "excluded_bad_meg_channels": [str(ch) for ch in excluded_bad_meg_channels],
            "noise_covariance": noise_cov_meta,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _find_inverse_operator_for_fif(self, fif_filename):
        """Find a likely precomputed inverse-operator FIF near the selected FIF."""
        if not fif_filename:
            return None
        fif_path = os.path.abspath(str(fif_filename))
        fif_dir = os.path.dirname(fif_path)
        parent_dir = os.path.dirname(fif_dir)
        inverse_dir = os.path.join(parent_dir, "inverse")

        candidates = []
        fif_stem = os.path.splitext(os.path.basename(fif_path))[0]
        for folder in [inverse_dir, fif_dir, parent_dir, os.path.join(parent_dir, "Anatomy")]:
            if not folder or not os.path.isdir(folder):
                continue
            candidates.extend(sorted(glob.glob(os.path.join(folder, "*-inv.fif"))))
            candidates.extend(sorted(glob.glob(os.path.join(folder, "*_inv.fif"))))
            candidates.extend(sorted(glob.glob(os.path.join(folder, "*inv*.fif"))))

        seen = []
        for path in candidates:
            p = os.path.abspath(path)
            if p not in seen:
                seen.append(p)
        if not seen:
            return None

        fif_stem = fif_stem.lower().replace("_raw", "").replace("-raw", "")

        def _name_matches_run_stem(name):
            base = str(name).lower()
            for suffix in ("-inv.fif", "_inv.fif"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            if base == fif_stem:
                return True
            for left in ("-", "_"):
                if base.startswith(f"{fif_stem}{left}") or base.endswith(f"{left}{fif_stem}"):
                    return True
                for right in ("-", "_"):
                    if f"{left}{fif_stem}{right}" in base:
                        return True
            return False

        preferred = [
            path for path in seen
            if _name_matches_run_stem(os.path.basename(path))
        ]
        for path in preferred:
            if self._inverse_operator_matches_fif(path, fif_filename):
                return path
        return None

    def _default_inverse_operator_output_path(self, fif_filename):
        """Return default output path for auto-generated inverse operator."""
        if not fif_filename:
            return None
        fif_path = os.path.abspath(str(fif_filename))
        fif_dir = os.path.dirname(fif_path)
        parent_dir = os.path.dirname(fif_dir)
        inverse_dir = os.path.join(parent_dir, "inverse")
        stem = os.path.splitext(os.path.basename(fif_path))[0]
        stem = stem.replace("_raw", "").replace("-raw", "")
        if stem.endswith("-inv"):
            out_name = f"{stem}.fif"
        else:
            out_name = f"{stem}-inv.fif"
        return os.path.join(inverse_dir, out_name)

    def _create_inverse_operator_for_fif(self, fif_filename, output_path=None):
        """Create and save inverse operator FIF when missing."""
        if not fif_filename:
            return None, "Cannot create inverse operator: FIF filename is missing."
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            return None, "Cannot create inverse operator: 3D viewer is unavailable."
        if not hasattr(self.meg_3d, "_get_inverse_for_meg_channels"):
            return None, "Cannot create inverse operator: MEG inverse helper is unavailable."

        cfg_ok, cfg_msg = self._auto_configure_freesurfer_subject_for_fif(
            fif_filename,
            require_bem=False,
        )
        if cfg_msg:
            self.c.log.emit(f"MEG source config: {cfg_msg}")
        if not cfg_ok:
            return None, f"Cannot create inverse operator: {cfg_msg}"

        if output_path is None or not str(output_path).strip():
            output_path = self._default_inverse_operator_output_path(fif_filename)
        if output_path is None:
            return None, "Cannot create inverse operator: output path could not be resolved."
        output_path = os.path.abspath(str(output_path))

        try:
            # Ensure MEG sensor info and head->MRI transform belong to this FIF.
            need_load = getattr(self.meg_3d, "_meg_sensor_info", None) is None
            if hasattr(self.meg_3d, "_fif_file_identity"):
                target_identity = self.meg_3d._fif_file_identity(fif_filename)
                current_identity = getattr(self.meg_3d, "_meg_current_fif_identity", None)
                need_load = need_load or target_identity != current_identity
            else:
                current_path = os.path.abspath(str(getattr(self.meg_3d, "_meg_current_fif_path", "") or ""))
                need_load = need_load or current_path != os.path.abspath(str(fif_filename))
            if need_load:
                loaded = bool(self.meg_3d.load_fif_sensors(fif_filename))
                if not loaded:
                    return None, "Cannot create inverse operator: failed to load FIF sensor info."

            meg_info = getattr(self.meg_3d, "_meg_sensor_info", None)
            if meg_info is None:
                return None, "Cannot create inverse operator: no MEG sensor info available."
            channel_names = list(meg_info.get("ch_names", []))
            if not channel_names:
                return None, "Cannot create inverse operator: no MEG channels available."

            self._reload_meg_transform_for_fif(fif_filename)

            noise_cov, noise_cov_meta = self._get_or_compute_noise_cov_for_fif(fif_filename)
            if noise_cov is None:
                return None, "Cannot create inverse operator: run-specific noise covariance is unavailable."
            if hasattr(self.meg_3d, "set_meg_noise_cov"):
                self.meg_3d.set_meg_noise_cov(noise_cov, metadata=noise_cov_meta)

            inv, _ = self.meg_3d._get_inverse_for_meg_channels(channel_names)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mne.minimum_norm.write_inverse_operator(output_path, inv, overwrite=True, verbose='error')
            self._write_inverse_operator_metadata(output_path, fif_filename, channel_names=channel_names)
            cov_label = str((noise_cov_meta or {}).get("selected_method") or (noise_cov_meta or {}).get("method") or "noise_cov")
            return output_path, f"Created inverse operator: {output_path} | noise_cov={cov_label}"
        except Exception as exc:
            return None, f"Failed to create inverse operator: {exc}"

    def _auto_configure_freesurfer_subject_for_fif(self, fif_filename, require_bem=True):
        """Try loading FreeSurfer subject from `../Anatomy` for FIF visualization/inverse."""
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            return False, "3D viewer unavailable."
        subjects_dir, subject_id, anatomy_dir = self._find_anatomy_subject_for_fif(
            fif_filename,
            require_bem=require_bem,
        )
        if not subjects_dir or not subject_id:
            if anatomy_dir and os.path.isdir(anatomy_dir):
                if require_bem:
                    return False, f"No FreeSurfer subject with surf+mri+bem found under {anatomy_dir}."
                return False, f"No FreeSurfer subject with surf+mri found under {anatomy_dir}."
            return False, "No Anatomy folder found for this FIF location."

        curr_subject = getattr(self.meg_3d, "custom_subject", None)
        curr_subjects_dir = getattr(self.meg_3d, "custom_subjects_dir", None)
        force_fsaverage = bool(getattr(self.meg_3d, "force_fsaverage", False))
        if (
            str(curr_subject) == str(subject_id)
            and curr_subjects_dir is not None
            and os.path.abspath(str(curr_subjects_dir)) == os.path.abspath(str(subjects_dir))
            and not force_fsaverage
        ):
            return True, f"Using FreeSurfer subject {subject_id} from {subjects_dir}."

        loaded = False
        if hasattr(self.meg_3d, "load_freesurfer_subject"):
            loaded = bool(self.meg_3d.load_freesurfer_subject(subject_id=subject_id, subjects_dir=subjects_dir))
        if not loaded:
            return False, f"Could not load FreeSurfer subject {subject_id} from {subjects_dir}."

        # Ensure viewer is showing the loaded subject (not forced fsaverage).
        if bool(getattr(self.meg_3d, "force_fsaverage", False)) and hasattr(self.meg_3d, "set_subject_view_mode"):
            try:
                self.meg_3d.set_subject_view_mode("subject")
            except Exception:
                pass
        elif hasattr(self.meg_3d, "_reload_brain_with_subject"):
            try:
                self.meg_3d._reload_brain_with_subject()
            except Exception:
                pass

        if hasattr(self.meg_3d, "_sync_subject_view_combo"):
            try:
                self.meg_3d._sync_subject_view_combo()
            except Exception:
                pass

        return True, f"Using FreeSurfer subject {subject_id} from {subjects_dir}."

    def _display_source_stc_for_meg_analysis(self, stc, time_sec):
        """Update the persistent 3D viewer to a specific source-estimate time."""
        if stc is None:
            return False
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            return False
        if not hasattr(self.meg_3d, "display_meg_stc_on_brain"):
            return False
        return bool(self.meg_3d.display_meg_stc_on_brain(stc, time_sec=float(time_sec)))

    def _set_source_brain_view_for_meg_analysis(self, mode):
        """Switch the persistent 3D viewer source display between patient and fsaverage."""
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            return False
        if not hasattr(self.meg_3d, "set_subject_view_mode"):
            return False
        mode = "fsaverage" if str(mode).strip().lower() == "fsaverage" else "subject"
        self.meg_3d.set_subject_view_mode(mode)
        return True

    @staticmethod
    def _source_bandpass_from_options(options):
        """Return source-analysis bandpass settings from UI options."""
        enabled = bool((options or {}).get("source_bandpass_enabled", False))
        if not enabled:
            return False, None, None, "unfiltered"
        try:
            l_freq = float((options or {}).get("bandpass_low_hz", 0.1))
        except Exception:
            l_freq = 0.1
        try:
            h_freq = float((options or {}).get("bandpass_high_hz", 150.0))
        except Exception:
            h_freq = 150.0
        l_freq = None if not np.isfinite(l_freq) or l_freq <= 0.0 else float(l_freq)
        h_freq = None if not np.isfinite(h_freq) or h_freq <= 0.0 else float(h_freq)
        if l_freq is None and h_freq is None:
            return False, None, None, "unfiltered"
        if l_freq is not None and h_freq is not None and h_freq <= l_freq:
            raise ValueError("Source bandpass high cutoff must be greater than low cutoff.")
        low_label = "0" if l_freq is None else f"{l_freq:g}"
        high_label = "Nyq" if h_freq is None else f"{h_freq:g}"
        return True, l_freq, h_freq, f"bandpass={low_label}-{high_label} Hz"

    @staticmethod
    def _apply_source_bandpass(inst, l_freq, h_freq):
        """Filter a copy of an Evoked/Epochs object for source estimation."""
        if inst is None or (l_freq is None and h_freq is None):
            return inst
        try:
            nyq = float(inst.info["sfreq"]) / 2.0
        except Exception:
            nyq = np.nan
        if h_freq is not None and np.isfinite(nyq) and float(h_freq) >= nyq:
            raise ValueError(
                f"Source bandpass high cutoff ({float(h_freq):g} Hz) must be below Nyquist "
                f"({nyq:g} Hz)."
            )
        filt = inst.copy()
        if hasattr(filt, "load_data"):
            try:
                filt.load_data(verbose="error")
            except TypeError:
                filt.load_data()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="filter_length.*is longer than the signal.*",
                category=RuntimeWarning,
            )
            filt.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks="meg",
                n_jobs=1,
                verbose="error",
            )
        return filt

    def _compute_source_stc_for_meg_analysis(self, options=None):
        """Compute source estimate from selected evoked for MEG analysis tab."""
        if options is None:
            options = {}
        if not getattr(self, "b_fif_loaded", False):
            return None, "No FIF loaded.", {}
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            return None, "3D viewer unavailable.", {}

        evoked_name = str(options.get("evoked_name", "") or "").strip()
        method = str(options.get("method", "MNE") or "MNE").strip()
        display_source = not bool(options.get("_defer_source_display", False))
        current_fif = os.path.abspath(str(getattr(self, "_fif_filename", "") or "")) if getattr(self, "_fif_filename", None) else ""

        def _normalize_source_fif(value):
            path = str(value or "").strip()
            if path in {"", "__current__"}:
                path = current_fif
            if not path:
                return ""
            return os.path.abspath(str(path))

        selected_fif_paths = []
        requested_paths = options.get("source_fif_paths", None)
        if isinstance(requested_paths, (list, tuple, np.ndarray)):
            for path in requested_paths:
                norm = _normalize_source_fif(path)
                if norm and norm not in selected_fif_paths:
                    selected_fif_paths.append(norm)
        if not selected_fif_paths:
            norm = _normalize_source_fif(options.get("source_fif_path", ""))
            if norm:
                selected_fif_paths.append(norm)
        if not selected_fif_paths:
            return None, "No source run selected.", {}

        if len(selected_fif_paths) > 1 and not bool(options.get("_source_multi_child", False)):
            epoch_indices_by_run = options.get("epoch_indices_by_run", {})
            if not isinstance(epoch_indices_by_run, dict):
                epoch_indices_by_run = {}
            normalized_epoch_indices_by_run = {}
            for key, values in epoch_indices_by_run.items():
                norm = _normalize_source_fif(key)
                if not norm:
                    continue
                try:
                    normalized_epoch_indices_by_run[norm] = [int(v) for v in list(values or [])]
                except Exception:
                    normalized_epoch_indices_by_run[norm] = []

            source_epoch_indices_explicit = bool(options.get("source_epoch_indices_explicit", False))
            stcs = []
            stc_paths = []
            per_run_meta = []
            skipped_runs = []
            for path in selected_fif_paths:
                run_epoch_indices = normalized_epoch_indices_by_run.get(path, [])
                if source_epoch_indices_explicit and not run_epoch_indices:
                    skipped_runs.append(os.path.basename(path))
                    continue
                child_options = dict(options)
                child_options.pop("source_fif_paths", None)
                child_options["source_fif_path"] = path
                child_options["source_run_label"] = os.path.basename(path)
                child_options["epoch_indices"] = list(run_epoch_indices)
                child_options["inverse_operator_path"] = ""
                child_options["_source_multi_child"] = True
                child_options["_defer_source_display"] = True
                child_result = self._compute_source_stc_for_meg_analysis(child_options)
                if not isinstance(child_result, tuple) or len(child_result) < 2:
                    return None, f"Multi-run source computation failed for {os.path.basename(path)}.", {
                        "source_fif_paths": selected_fif_paths,
                    }
                stc, child_msg = child_result[0], str(child_result[1])
                child_meta = child_result[2] if len(child_result) >= 3 and isinstance(child_result[2], dict) else {}
                if stc is None:
                    return None, f"Multi-run source computation failed for {os.path.basename(path)}: {child_msg}", {
                        "source_fif_paths": selected_fif_paths,
                        "failed_run": path,
                        "per_run_meta": per_run_meta,
                    }
                stcs.append(stc)
                stc_paths.append(path)
                per_run_meta.append(dict(child_meta))

            if not stcs:
                return None, "No selected source runs had epochs for the selected stimulus.", {
                    "source_fif_paths": selected_fif_paths,
                    "skipped_runs": skipped_runs,
                }

            ref = stcs[0]
            ref_times = np.asarray(getattr(ref, "times", []), dtype=float)
            ref_shape = np.asarray(ref.data).shape
            ref_vertices = getattr(ref, "vertices", None)
            data_stack = []
            for idx, stc in enumerate(stcs):
                data = np.asarray(stc.data, dtype=float)
                if data.shape != ref_shape:
                    return None, (
                        "Selected runs produced incompatible source estimate shapes; "
                        f"{os.path.basename(stc_paths[idx])} has {data.shape}, expected {ref_shape}."
                    ), {"source_fif_paths": selected_fif_paths, "per_run_meta": per_run_meta}
                times = np.asarray(getattr(stc, "times", []), dtype=float)
                if times.shape != ref_times.shape or not np.allclose(times, ref_times, rtol=1e-6, atol=1e-9):
                    return None, (
                        "Selected runs produced incompatible source time axes; "
                        f"{os.path.basename(stc_paths[idx])} cannot be averaged with the first run."
                    ), {"source_fif_paths": selected_fif_paths, "per_run_meta": per_run_meta}
                vertices = getattr(stc, "vertices", None)
                if ref_vertices is not None and vertices is not None:
                    if len(ref_vertices) != len(vertices) or any(
                        not np.array_equal(np.asarray(a), np.asarray(b))
                        for a, b in zip(ref_vertices, vertices)
                    ):
                        return None, (
                            "Selected runs produced incompatible source vertices; "
                            f"{os.path.basename(stc_paths[idx])} cannot be averaged with the first run."
                        ), {"source_fif_paths": selected_fif_paths, "per_run_meta": per_run_meta}
                data_stack.append(data)

            avg_stc = ref.copy()
            avg_data = np.nanmean(np.stack(data_stack, axis=0), axis=0)
            try:
                avg_stc._data = avg_data
            except Exception:
                avg_stc.data[:] = avg_data
            try:
                initial_time = float(options.get("initial_time", 0.10))
            except Exception:
                initial_time = 0.10
            if hasattr(self.meg_3d, "_meg_last_stc"):
                self.meg_3d._meg_last_stc = avg_stc.copy()
            if display_source and hasattr(self.meg_3d, "display_meg_stc_on_brain"):
                try:
                    self.meg_3d.display_meg_stc_on_brain(avg_stc, time_sec=initial_time)
                except Exception:
                    pass
            try:
                lambda2_for_msg = float(options.get("lambda2", 1.0 / 9.0))
            except Exception:
                lambda2_for_msg = 1.0 / 9.0
            try:
                multi_bandpass_enabled, _multi_l_freq, _multi_h_freq, multi_filter_label = self._source_bandpass_from_options(options)
            except Exception:
                multi_bandpass_enabled, multi_filter_label = False, "unfiltered"
            multi_filter_msg = f", {multi_filter_label}" if multi_bandpass_enabled else ""
            selected_epoch_total = int(sum(int(meta.get("n_selected_epochs", 0) or 0) for meta in per_run_meta))
            meta = {
                "source_fif_paths": list(selected_fif_paths),
                "computed_source_fif_paths": list(stc_paths),
                "source_fif_path": "",
                "source_run_label": str(options.get("source_run_label", "") or f"{len(stcs)} selected runs"),
                "n_source_runs": int(len(stcs)),
                "n_requested_source_runs": int(len(selected_fif_paths)),
                "n_selected_epochs": selected_epoch_total,
                "skipped_runs": skipped_runs,
                "per_run_meta": per_run_meta,
                "source_src": per_run_meta[0].get("source_src") if per_run_meta else None,
            }
            return avg_stc, (
                f"Source computed by averaging source estimates across {len(stcs)} selected runs "
                f"({method}, lambda2={lambda2_for_msg:.6f}, selected_epochs={selected_epoch_total}{multi_filter_msg})."
            ), meta

        selected_fif = selected_fif_paths[0]
        use_current_run = bool(current_fif and selected_fif == current_fif)

        run_raw = getattr(self, "_fif_raw", None) if use_current_run else None
        if run_raw is None:
            try:
                run_raw = mne.io.read_raw_fif(selected_fif, preload=False, verbose="error")
            except Exception as exc:
                return None, f"Could not load selected source run: {exc}", {"source_fif_path": selected_fif}

        run_epochs = None
        if hasattr(self, "_fif_epochs_by_path") and isinstance(self._fif_epochs_by_path, dict):
            run_epochs = self._fif_epochs_by_path.get(selected_fif)
        if run_epochs is None and use_current_run:
            run_epochs = getattr(self, "_fif_epochs", None)
        if run_epochs is None:
            try:
                run_epochs, run_meta = self._compute_topomap_epochs_for_fif(selected_fif, raw=run_raw)
                if not hasattr(self, "_fif_epochs_by_path") or not isinstance(self._fif_epochs_by_path, dict):
                    self._fif_epochs_by_path = {}
                if not hasattr(self, "_fif_epoch_metadata_by_path") or not isinstance(self._fif_epoch_metadata_by_path, dict):
                    self._fif_epoch_metadata_by_path = {}
                self._fif_epochs_by_path[selected_fif] = run_epochs
                self._fif_epoch_metadata_by_path[selected_fif] = dict(run_meta or {})
            except Exception as exc:
                return None, f"Could not prepare selected run epochs: {exc}", {"source_fif_path": selected_fif}
        if run_epochs is None or len(run_epochs) == 0:
            return None, "No epochs available for selected source run.", {"source_fif_path": selected_fif}

        try:
            initial_time = float(options.get("initial_time", 0.10))
        except Exception:
            initial_time = 0.10
        try:
            lambda2 = float(options.get("lambda2", 1.0 / 9.0))
        except Exception:
            lambda2 = 1.0 / 9.0
        lambda2 = max(lambda2, 1e-12)
        source_space_kind = str(options.get("source_space_kind", "surface") or "surface").strip().lower()
        if source_space_kind not in {"surface", "volume"}:
            source_space_kind = "surface"
        if hasattr(self.meg_3d, "_meg_source_space_kind"):
            self.meg_3d._meg_source_space_kind = source_space_kind
        source_space_combo = getattr(self.meg_3d, "meg_source_space_combo", None)
        if source_space_combo is not None:
            try:
                source_space_combo.blockSignals(True)
                idx = source_space_combo.findData(source_space_kind)
                if idx >= 0:
                    source_space_combo.setCurrentIndex(idx)
                source_space_combo.blockSignals(False)
            except Exception:
                try:
                    source_space_combo.blockSignals(False)
                except Exception:
                    pass
        inverse_operator_path = str(options.get("inverse_operator_path", "") or "").strip()
        epoch_indices = options.get("epoch_indices", [])
        if not isinstance(epoch_indices, (list, tuple, np.ndarray)):
            epoch_indices = []
        try:
            epoch_indices = [int(i) for i in epoch_indices]
        except Exception:
            epoch_indices = []

        evoked = None
        valid_epoch_indices = [
            int(i)
            for i in epoch_indices
            if 0 <= int(i) < int(len(run_epochs))
        ]
        if bool(options.get("source_epoch_indices_explicit", False)) and not valid_epoch_indices:
            return None, "No selected stimulus epochs found for selected source run.", {"source_fif_path": selected_fif}
        if valid_epoch_indices:
            try:
                evoked = run_epochs[valid_epoch_indices].average()
            except Exception:
                pass
        if evoked is None and evoked_name and use_current_run and isinstance(getattr(self, "_fif_evokeds", {}), dict):
            evoked_from_map = self._fif_evokeds.get(evoked_name)
            if evoked_from_map is not None:
                evoked = evoked_from_map
        if evoked is None:
            try:
                evoked = run_epochs.average()
            except Exception as exc:
                return None, f"Could not average selected run epochs: {exc}", {"source_fif_path": selected_fif}

        try:
            source_bandpass_enabled, source_l_freq, source_h_freq, source_filter_label = self._source_bandpass_from_options(options)
        except Exception as exc:
            return None, str(exc), {"source_fif_path": selected_fif}
        epochs_for_lcmv = None
        if source_bandpass_enabled:
            try:
                evoked = self._apply_source_bandpass(evoked, source_l_freq, source_h_freq)
                epochs_for_lcmv = run_epochs[valid_epoch_indices] if valid_epoch_indices else run_epochs
                epochs_for_lcmv = self._apply_source_bandpass(epochs_for_lcmv, source_l_freq, source_h_freq)
            except Exception as exc:
                return None, f"Could not apply source bandpass filter: {exc}", {"source_fif_path": selected_fif}
        else:
            epochs_for_lcmv = run_epochs[valid_epoch_indices] if valid_epoch_indices else run_epochs

        cfg_ok, cfg_msg = self._auto_configure_freesurfer_subject_for_fif(
            selected_fif,
            require_bem=False,
        )
        if cfg_msg:
            self.c.log.emit(f"MEG source config: {cfg_msg}")

        meta = {
            "source_fif_path": selected_fif,
            "source_fif": os.path.basename(selected_fif),
            "source_run_label": str(options.get("source_run_label", "") or os.path.basename(selected_fif)),
            "run_id": self._infer_run_id_from_fif_filename(selected_fif),
            "n_source_epochs": int(len(run_epochs)),
            "n_selected_epochs": int(len(valid_epoch_indices)) if valid_epoch_indices else int(len(run_epochs)),
            "source_space_kind": source_space_kind,
            "source_bandpass_enabled": bool(source_bandpass_enabled),
            "source_bandpass_label": str(source_filter_label),
            "source_bandpass_low_hz": source_l_freq,
            "source_bandpass_high_hz": source_h_freq,
        }
        if hasattr(self, "meg_3d") and self.meg_3d is not None and hasattr(self.meg_3d, "_get_brain_subject_info"):
            try:
                subject_id, subjects_dir = self.meg_3d._get_brain_subject_info()
                meta["subject"] = str(subject_id)
                meta["subjects_dir"] = str(subjects_dir)
            except Exception:
                pass
        if not use_current_run and hasattr(self.meg_3d, "load_fif_sensors"):
            try:
                self.meg_3d.load_fif_sensors(selected_fif)
            except Exception as exc:
                meta["source_sensor_load_warning"] = str(exc)
        try:
            noise_cov, noise_cov_meta = self._get_or_compute_noise_cov_for_fif(
                selected_fif,
                raw=run_raw,
            )
            if hasattr(self.meg_3d, "set_meg_noise_cov"):
                self.meg_3d.set_meg_noise_cov(noise_cov, metadata=noise_cov_meta)
            try:
                self._fif_noise_cov = noise_cov.copy()
            except Exception:
                self._fif_noise_cov = noise_cov
            self._fif_noise_cov_metadata = dict(noise_cov_meta or {})
            meta["noise_cov_method"] = str((noise_cov_meta or {}).get("method", ""))
            meta["noise_cov_selected_method"] = str((noise_cov_meta or {}).get("selected_method", ""))
            meta["noise_cov_kind"] = str((noise_cov_meta or {}).get("kind", ""))
            if (noise_cov_meta or {}).get("kind") == "raw_fallback":
                meta["noise_cov_warning"] = str((noise_cov_meta or {}).get("fallback_reason", ""))
        except Exception as exc:
            meta["noise_cov_method"] = ""
            meta["noise_cov_error"] = str(exc)

        if method.upper() == "LCMV":
            try:
                stc = self._compute_lcmv_stc_from_evoked(evoked, epochs_for_lcmv)
                if hasattr(self.meg_3d, "set_meg_source_estimate_context"):
                    self.meg_3d.set_meg_source_estimate_context(
                        stc,
                        src=getattr(self.meg_3d, "_meg_source_src", None),
                    )
                meta["source_src"] = getattr(self.meg_3d, "_meg_source_src", None)
                if hasattr(self.meg_3d, "_meg_last_stc"):
                    self.meg_3d._meg_last_stc = stc.copy()
                if display_source and hasattr(self.meg_3d, "display_meg_stc_on_brain"):
                    try:
                        self.meg_3d.display_meg_stc_on_brain(stc, time_sec=initial_time)
                    except Exception:
                        pass
                if valid_epoch_indices:
                    return stc, (
                        f"Source computed from selected stimulus epochs "
                        f"(run={os.path.basename(selected_fif)}, n={len(valid_epoch_indices)}, LCMV"
                        f"{', ' + source_filter_label if source_bandpass_enabled else ''})."
                    ), meta
                return stc, (
                    f"Source computed using LCMV beamformer (run={os.path.basename(selected_fif)}"
                    f"{', ' + source_filter_label if source_bandpass_enabled else ''})."
                ), meta
            except Exception as exc:
                return None, f"LCMV source computation failed: {exc}", meta

        if not inverse_operator_path:
            auto_inv = self._find_inverse_operator_for_fif(selected_fif)
            if auto_inv and os.path.isfile(auto_inv):
                inverse_operator_path = auto_inv
            else:
                created_path, create_msg = self._create_inverse_operator_for_fif(
                    selected_fif,
                    output_path=self._default_inverse_operator_output_path(selected_fif),
                )
                if create_msg:
                    self.c.log.emit(create_msg)
                if created_path and os.path.isfile(created_path):
                    inverse_operator_path = created_path

        if inverse_operator_path:
            if not os.path.isfile(inverse_operator_path):
                return None, f"Inverse operator not found: {inverse_operator_path}", meta
            if not self._inverse_operator_matches_fif(
                inverse_operator_path,
                selected_fif,
            ):
                stale_inverse_path = inverse_operator_path
                created_path, create_msg = self._create_inverse_operator_for_fif(
                    selected_fif,
                    output_path=self._default_inverse_operator_output_path(selected_fif),
                )
                if create_msg:
                    self.c.log.emit(create_msg)
                if not created_path or not os.path.isfile(created_path):
                    return None, (
                        "Inverse operator does not match the current FIF/transform metadata, "
                        f"and automatic regeneration failed. {create_msg or ''}"
                    ), meta
                if not self._inverse_operator_matches_fif(
                    created_path,
                    selected_fif,
                ):
                    return None, (
                        "Regenerated inverse operator still does not match the current FIF/transform metadata. "
                        "Reload the FIF after saving the transform, then compute source time course again."
                    ), meta
                inverse_operator_path = created_path
                meta["stale_inverse_operator_path"] = str(stale_inverse_path)
                meta["inverse_operator_path"] = str(inverse_operator_path)
                meta["inverse_operator_regenerated"] = True
            try:
                inv = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose='error')
                evoked_meg = evoked.copy().pick("meg", exclude="bads")
                stc = mne.minimum_norm.apply_inverse(
                    evoked_meg,
                    inv,
                    lambda2=lambda2,
                    method=method,
                    verbose='error',
                )
                if hasattr(self.meg_3d, "set_meg_source_estimate_context"):
                    self.meg_3d.set_meg_source_estimate_context(
                        stc,
                        src=inv.get("src"),
                        subject=getattr(stc, "subject", None),
                    )
                if hasattr(self.meg_3d, "_meg_last_stc"):
                    self.meg_3d._meg_last_stc = stc.copy()
                if display_source and hasattr(self.meg_3d, "display_meg_stc_on_brain"):
                    try:
                        self.meg_3d.display_meg_stc_on_brain(stc, time_sec=initial_time)
                    except Exception:
                        pass
                meta["inverse_operator_path"] = str(inverse_operator_path)
                meta["source_src"] = inv.get("src")
                if valid_epoch_indices:
                    return stc, (
                        f"Source computed from selected stimulus epochs "
                        f"(run={os.path.basename(selected_fif)}, n={len(valid_epoch_indices)}, {method}, "
                        f"lambda2={lambda2:.6f}{', ' + source_filter_label if source_bandpass_enabled else ''})."
                    ), meta
                return stc, (
                    f"Source computed using inverse file (run={os.path.basename(selected_fif)}, {method}, "
                    f"lambda2={lambda2:.6f}{', ' + source_filter_label if source_bandpass_enabled else ''})."
                ), meta
            except Exception as exc:
                return None, f"Source computation with inverse file failed: {exc}", meta

        try:
            if not use_current_run and hasattr(self.meg_3d, "load_fif_sensors"):
                try:
                    self.meg_3d.load_fif_sensors(selected_fif)
                except Exception:
                    pass
            projected = bool(
                self.meg_3d.project_meg_evoked_to_source(
                    evoked,
                    method=method,
                    lambda2=lambda2,
                )
            )
        except Exception as exc:
            return None, f"Source projection failed: {exc}", meta
        if not projected:
            if not cfg_ok:
                return None, f"Projection failed; {cfg_msg}", meta
            return None, "Projection failed.", meta

        stc = None
        if hasattr(self.meg_3d, "get_last_meg_source_estimate"):
            stc = self.meg_3d.get_last_meg_source_estimate(copy=True)
        if stc is None:
            return None, "Projection completed but no source estimate is available.", meta
        meta["source_src"] = getattr(self.meg_3d, "_meg_last_stc_src", None)
        if display_source and hasattr(self.meg_3d, "display_meg_stc_on_brain"):
            try:
                self.meg_3d.display_meg_stc_on_brain(stc, time_sec=initial_time)
            except Exception:
                pass
        return stc, f"Inverse source computed (run={os.path.basename(selected_fif)}, {method}, lambda2={lambda2:.6f}).", meta




    def _trial_df_for_loaded_fif(self):
        trial_df = getattr(self, "_fif_trial_response_table", pd.DataFrame())
        if not isinstance(trial_df, pd.DataFrame) or trial_df.empty:
            return pd.DataFrame()
        df = trial_df.copy()
        if "source_fif_path" in df.columns and getattr(self, "_fif_filename", None):
            current = os.path.abspath(str(self._fif_filename))
            mask = df["source_fif_path"].astype(str) == current
            if np.any(mask):
                df = df.loc[mask].copy()
        df = self._annotate_trial_df_with_stimulus_groups(df, getattr(self, "_fif_filename", None))
        return df.reset_index(drop=True)

    def _stimulus_labels_for_loaded_epochs(self, group_mode="canonical"):
        epochs = getattr(self, "_fif_epochs", None)
        if epochs is None or len(epochs) == 0:
            return [], pd.DataFrame()
        trial_df = self._trial_df_for_loaded_fif()
        if trial_df.empty:
            return [], pd.DataFrame()
        n = min(len(epochs), len(trial_df))
        trial_df = trial_df.iloc[:n].copy().reset_index(drop=True)
        labels = []
        for i in range(n):
            row = trial_df.iloc[i]
            if str(group_mode) == "individual":
                stim = row.get("stimulus")
                labels.append(os.path.basename(str(stim))) if pd.notna(stim) else labels.append("unknown")
            elif str(group_mode) == "category":
                stim_group = str(row.get("stimulus_group", "") or "").strip()
                if stim_group:
                    labels.append(stim_group)
                else:
                    canonical = row.get("canonical_stimulus")
                    if pd.notna(canonical) and str(canonical).strip():
                        labels.append(str(canonical))
                    else:
                        labels.append(self._canonicalize_meg_stimulus_label(row.get("stimulus")))
            else:
                canonical = row.get("canonical_stimulus")
                if pd.notna(canonical) and str(canonical).strip():
                    labels.append(str(canonical))
                else:
                    labels.append(self._canonicalize_meg_stimulus_label(row.get("stimulus")))
        return labels, trial_df

    @staticmethod
    def _ctx_volume_name_to_annot_name(label_name):
        text = str(label_name or "").strip()
        if text.lower().startswith("ctx-lh-"):
            return f"{text[7:]}-lh"
        if text.lower().startswith("ctx-rh-"):
            return f"{text[7:]}-rh"
        return text

    def _default_semantic_roi_names(self):
        roi_names = []
        try:
            if hasattr(self, "meg_3d") and self.meg_3d is not None:
                active = getattr(self.meg_3d, "active_labels", [])
                if isinstance(active, (list, tuple)):
                    roi_names = [str(v) for v in active if str(v).lower().startswith("ctx-")]
        except Exception:
            roi_names = []
        if roi_names:
            return list(dict.fromkeys(roi_names))
        return [
            "ctx-lh-bankssts",
            "ctx-lh-inferiortemporal",
            "ctx-lh-superiortemporal",
            "ctx-lh-temporalpole",
            "ctx-lh-transversetemporal",
            "ctx-lh-insula",
            "ctx-lh-parahippocampal",
        ]

    @staticmethod
    def _annot_name_to_ctx_volume_name(label_name):
        text = str(label_name or "").strip()
        if text.endswith("-lh"):
            return f"ctx-lh-{text[:-3]}"
        if text.endswith("-rh"):
            return f"ctx-rh-{text[:-3]}"
        return text

    def _get_available_semantic_parcellations(self, subject=None, subjects_dir=None):
        if not subject or not subjects_dir:
            return self._default_semantic_roi_names()
        try:
            annot_labels = mne.read_labels_from_annot(
                subject=str(subject),
                parc="aparc",
                subjects_dir=str(subjects_dir),
                verbose="error",
            )
        except Exception:
            return self._default_semantic_roi_names()
        names = []
        for lbl in annot_labels:
            ctx_name = self._annot_name_to_ctx_volume_name(getattr(lbl, "name", ""))
            lname = ctx_name.lower()
            if not lname.startswith("ctx-") or lname.endswith("unknown"):
                continue
            names.append(ctx_name)
        names = sorted(set(names))
        return names or self._default_semantic_roi_names()

    def _get_forward_and_noise_for_meg_info(self, info):
        if info is None:
            raise RuntimeError("MEG info is unavailable.")
        if not hasattr(self, "meg_3d") or self.meg_3d is None:
            raise RuntimeError("3D viewer is unavailable for forward modeling.")
        if not hasattr(self.meg_3d, "_get_source_space_and_bem") or not hasattr(self.meg_3d, "_resolve_meg_sensor_trans"):
            raise RuntimeError("3D viewer source-model helpers are unavailable.")

        ch_names = list(getattr(info, "ch_names", info["ch_names"]))
        src, bem = self.meg_3d._get_source_space_and_bem()
        trans, _trans_name = self.meg_3d._resolve_meg_sensor_trans()
        try:
            fwd = mne.make_forward_solution(
                info,
                trans=trans,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                ignore_ref=True,
                n_jobs=1,
                verbose="error",
            )
        except RuntimeError as exc:
            if "inside the scalp surface" not in str(exc):
                raise
            fwd = mne.make_forward_solution(
                info,
                trans=trans,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                ignore_ref=True,
                on_inside="warn",
                n_jobs=1,
                verbose="error",
            )
            self.c.log.emit("Forward model retry: proceeding with on_inside='warn' due to sensor/scalp mismatch.")

        noise_cov = None
        if getattr(self, "_fif_noise_cov", None) is not None:
            try:
                noise_cov = self._fif_noise_cov.copy().pick_channels(ch_names, ordered=True)
            except Exception:
                try:
                    noise_cov = mne.pick_channels_cov(
                        self._fif_noise_cov,
                        include=ch_names,
                        exclude=[],
                        ordered=True,
                        copy=True,
                    )
                except Exception:
                    noise_cov = None
        if noise_cov is None:
            try:
                current_fif = getattr(self, "_fif_filename", None)
                if current_fif:
                    run_cov, run_cov_meta = self._get_or_compute_noise_cov_for_fif(
                        current_fif,
                        raw=getattr(self, "_fif_raw", None),
                    )
                    self._fif_noise_cov_metadata = dict(run_cov_meta or {})
                    noise_cov = run_cov.copy().pick_channels(ch_names, ordered=True)
                    if hasattr(self.meg_3d, "set_meg_noise_cov"):
                        self.meg_3d.set_meg_noise_cov(run_cov, metadata=run_cov_meta)
            except Exception:
                noise_cov = None
        if noise_cov is None:
            if getattr(self, "_fif_noise_cov", None) is not None or bool(getattr(self, "_fif_noise_cov_metadata", {}) or {}):
                raise RuntimeError(
                    "Run-specific MEG noise covariance could not be aligned to the selected channels."
                )
            noise_cov = mne.make_ad_hoc_cov(info, verbose="error")
        return fwd, noise_cov

    @staticmethod
    def _response_cov_window(epochs):
        if epochs is None:
            return 0.0, None
        tmin = max(0.0, float(getattr(epochs, "tmin", 0.0)))
        tmax = float(getattr(epochs, "tmax", 0.0))
        if tmax <= tmin:
            tmax = None
        return tmin, tmax

    def _compute_data_cov_for_epochs(self, epochs):
        if epochs is None or len(epochs) == 0:
            raise RuntimeError("No epochs available for beamformer covariance.")
        epochs_meg = epochs.copy().pick("meg", exclude="bads")
        cov_tmin, cov_tmax = self._response_cov_window(epochs_meg)
        return mne.compute_covariance(
            epochs_meg,
            tmin=cov_tmin,
            tmax=cov_tmax,
            method="empirical",
            rank="info",
            verbose="error",
        )

    def _compute_lcmv_stc_from_evoked(self, evoked, epochs_for_cov):
        if evoked is None:
            raise RuntimeError("Evoked data is unavailable for LCMV.")
        if epochs_for_cov is None or len(epochs_for_cov) == 0:
            raise RuntimeError("Epochs are unavailable for LCMV covariance.")
        evoked_meg = evoked.copy().pick("meg", exclude="bads")
        fwd, noise_cov = self._get_forward_and_noise_for_meg_info(evoked_meg.info)
        data_cov = self._compute_data_cov_for_epochs(epochs_for_cov)
        filters = mne.beamformer.make_lcmv(
            evoked_meg.info,
            fwd,
            data_cov,
            reg=0.05,
            noise_cov=noise_cov,
            pick_ori=None,
            rank="info",
            weight_norm="unit-noise-gain-invariant",
            verbose="error",
        )
        stc = mne.beamformer.apply_lcmv(evoked_meg, filters, verbose="error")
        return stc

    def _compute_lcmv_stcs_from_epochs(self, epochs):
        if epochs is None or len(epochs) == 0:
            raise RuntimeError("No epochs available for LCMV.")
        epochs_meg = epochs.copy().pick("meg", exclude="bads")
        fwd, noise_cov = self._get_forward_and_noise_for_meg_info(epochs_meg.info)
        data_cov = self._compute_data_cov_for_epochs(epochs_meg)
        filters = mne.beamformer.make_lcmv(
            epochs_meg.info,
            fwd,
            data_cov,
            reg=0.05,
            noise_cov=noise_cov,
            pick_ori=None,
            rank="info",
            weight_norm="unit-noise-gain-invariant",
            verbose="error",
        )
        return mne.beamformer.apply_lcmv_epochs(
            epochs_meg,
            filters,
            return_generator=False,
            verbose="error",
        )

    @staticmethod
    def _mask_stc_to_labels(stc, labels):
        if stc is None or not labels:
            return stc
        try:
            stc_out = stc.copy()
            data = np.asarray(stc_out.data, dtype=float)
            if data.ndim != 2:
                return stc_out
            keep = np.zeros(data.shape[0], dtype=bool)
            lh_vertices = np.asarray(stc_out.vertices[0], dtype=int) if len(stc_out.vertices) > 0 else np.asarray([], dtype=int)
            rh_vertices = np.asarray(stc_out.vertices[1], dtype=int) if len(stc_out.vertices) > 1 else np.asarray([], dtype=int)
            for lbl in labels:
                hemi = getattr(lbl, "hemi", "")
                lbl_vertices = np.asarray(getattr(lbl, "vertices", []), dtype=int)
                if hemi == "lh" and lh_vertices.size > 0:
                    keep[: lh_vertices.size] |= np.isin(lh_vertices, lbl_vertices)
                elif hemi == "rh" and rh_vertices.size > 0:
                    keep[lh_vertices.size : lh_vertices.size + rh_vertices.size] |= np.isin(rh_vertices, lbl_vertices)
            data_masked = data.copy()
            data_masked[~keep, :] = 0.0
            stc_out._data = data_masked
            return stc_out
        except Exception:
            return stc

    def _compute_parcel_trial_heatmaps_for_meg_analysis(self, options=None):
        """Extract trial-level source parcel time courses for the heatmap tab."""
        if options is None:
            options = {}
        if not getattr(self, "b_fif_loaded", False):
            return None, "No FIF loaded."
        return build_source_parcel_trial_time_courses(self, options)

    def _compute_concept_decoding_for_meg_analysis(self, options=None):
        """Train/test concept decoder using all mapped MEG FIF runs."""
        if options is None:
            options = {}
        decoder_type = str(options.get("decoder_type", "logistic") or "logistic").strip().lower()
        if decoder_type in {
            "transformer",
            "axial_transformer",
            "registered_transformer",
            "embedding_mlp",
            "mlp_embedding",
            "word2vec_mlp",
            "neural_embedding",
        }:
            transformer_options = dict(options)
            transformer_options["feature_space"] = "signal"
            return compute_signal_space_concept_decoding(self, transformer_options)
        feature_space = str(options.get("feature_space", "signal") or "signal").strip().lower()
        if feature_space in {"source", "source_space", "source-space", "roi", "parcels"}:
            return compute_source_space_concept_decoding(self, options)
        return compute_signal_space_concept_decoding(self, options)

    def _compute_semantic_preview_for_meg_analysis(self, options=None):
        """Compute window-averaged source preview for one stimulus group."""
        if options is None:
            options = {}
        if not getattr(self, "b_fif_loaded", False):
            return None, "No FIF loaded."
        epochs = getattr(self, "_fif_epochs", None)
        if epochs is None or len(epochs) == 0:
            return None, "No epochs available for semantic preview."

        group_mode = str(options.get("group_mode", "canonical") or "canonical").strip().lower()
        if group_mode not in {"canonical", "category", "individual"}:
            group_mode = "canonical"
        stimulus_name = str(options.get("stimulus_name", "All stimuli") or "All stimuli").strip()
        trial_epoch_index = options.get("trial_epoch_index", None)
        try:
            trial_epoch_index = None if trial_epoch_index in ("", None) else int(trial_epoch_index)
        except Exception:
            trial_epoch_index = None
        try:
            window_start = float(options.get("window_start", 0.10))
            window_end = float(options.get("window_end", 0.30))
        except Exception:
            return None, "Invalid semantic preview window."
        if not np.isfinite(window_start) or not np.isfinite(window_end) or window_end <= window_start:
            return None, "Semantic preview requires a valid time window."
        method = str(options.get("method", "MNE") or "MNE").strip()
        try:
            lambda2 = float(options.get("lambda2", 1.0 / 9.0))
        except Exception:
            lambda2 = 1.0 / 9.0
        lambda2 = max(lambda2, 1e-12)
        inverse_operator_path = str(options.get("inverse_operator_path", "") or "").strip()
        requested_parcellations = options.get("selected_parcellations", [])
        if not isinstance(requested_parcellations, (list, tuple, np.ndarray)):
            requested_parcellations = []
        requested_parcellations = [str(v) for v in requested_parcellations if str(v).strip()]

        labels_all, trial_df = self._stimulus_labels_for_loaded_epochs(group_mode=group_mode)
        if not labels_all:
            return None, "No stimulus labels available for semantic preview."
        n = min(len(epochs), len(labels_all))
        epochs = epochs[:n]
        labels_all = [str(v) for v in labels_all[:n]]
        valid_mask = np.asarray([bool(str(v).strip()) and str(v).strip().lower() != "unknown" for v in labels_all], dtype=bool)
        epochs = epochs[valid_mask]
        labels_all = [labels_all[i] for i in range(len(labels_all)) if valid_mask[i]]
        if not labels_all:
            return None, "No valid stimulus labels available for semantic preview."
        available_stimulus_groups = list(pd.Index(labels_all).drop_duplicates())
        if stimulus_name == "All stimuli":
            selected_idx = list(range(len(labels_all)))
            preview_label = "All stimuli"
        else:
            selected_idx = [i for i, lbl in enumerate(labels_all) if lbl == stimulus_name]
            preview_label = stimulus_name
        if not selected_idx:
            return None, f"No trials matched stimulus selection: {stimulus_name}"
        selected_idx = [int(i) for i in selected_idx]
        if trial_epoch_index is not None:
            if trial_epoch_index not in selected_idx:
                return None, f"Selected trial {trial_epoch_index} is not part of stimulus {preview_label}."
            selected_idx = [int(trial_epoch_index)]

        cfg_ok, cfg_msg = self._auto_configure_freesurfer_subject_for_fif(
            getattr(self, "_fif_filename", None),
            require_bem=False,
        )
        if cfg_msg:
            self.c.log.emit(f"MEG semantic preview config: {cfg_msg}")

        subject = None
        subjects_dir = None
        if hasattr(self, "meg_3d") and self.meg_3d is not None and hasattr(self.meg_3d, "_get_brain_subject_info"):
            try:
                subject, subjects_dir = self.meg_3d._get_brain_subject_info()
            except Exception:
                subject, subjects_dir = None, None
        if not subject or not subjects_dir:
            return None, "No FreeSurfer subject is available for semantic preview."

        if not inverse_operator_path:
            auto_inv = self._find_inverse_operator_for_fif(getattr(self, "_fif_filename", None))
            if auto_inv and os.path.isfile(auto_inv):
                inverse_operator_path = auto_inv
            else:
                created_path, create_msg = self._create_inverse_operator_for_fif(
                    getattr(self, "_fif_filename", None),
                    output_path=self._default_inverse_operator_output_path(getattr(self, "_fif_filename", None)),
                )
                if create_msg:
                    self.c.log.emit(create_msg)
                if created_path and os.path.isfile(created_path):
                    inverse_operator_path = created_path
        if not inverse_operator_path or not os.path.isfile(inverse_operator_path):
            return None, "Inverse operator is required for semantic preview."

        available_parcellations = self._get_available_semantic_parcellations(subject=subject, subjects_dir=subjects_dir)
        try:
            annot_labels = mne.read_labels_from_annot(
                subject=str(subject),
                parc="aparc",
                subjects_dir=str(subjects_dir),
                verbose="error",
            )
        except Exception as exc:
            return None, f"Could not load aparc labels for semantic preview: {exc}"
        annot_map = {str(lbl.name): lbl for lbl in annot_labels}
        selected_roi_names = [name for name in requested_parcellations if name in set(available_parcellations)]
        selected_labels = []
        for roi_name in selected_roi_names:
            lbl = annot_map.get(str(self._ctx_volume_name_to_annot_name(roi_name)))
            if lbl is not None:
                selected_labels.append(lbl)

        try:
            evoked = epochs[selected_idx].average().copy().pick("meg", exclude="bads")
            if method.upper() == "LCMV":
                stc = self._compute_lcmv_stc_from_evoked(evoked, epochs[selected_idx])
            else:
                inv = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose="error")
                stc = mne.minimum_norm.apply_inverse(
                    evoked,
                    inv,
                    lambda2=lambda2,
                    method=method,
                    verbose="error",
                )
        except Exception as exc:
            return None, f"Semantic preview source computation failed: {exc}"

        times = np.asarray(stc.times, dtype=float)
        window_mask = (times >= window_start) & (times <= window_end)
        if not np.any(window_mask):
            return None, "Semantic preview window does not overlap the source time axis."

        try:
            selected_tc = np.asarray(
                mne.extract_label_time_course(
                    [stc],
                    selected_labels,
                    inv["src"],
                    mode="mean_flip",
                    return_generator=False,
                    allow_empty=True,
                    verbose="error",
                )[0],
                dtype=float,
            ) if selected_labels else np.empty((0, len(times)), dtype=float)
        except Exception:
            selected_tc = np.empty((0, len(times)), dtype=float)
        whole_brain_tc = np.nanmean(np.abs(np.asarray(stc.data, dtype=float)), axis=0)
        mean_vertex = np.nanmean(np.abs(np.asarray(stc.data[:, window_mask], dtype=float)), axis=1)
        window_stc = stc.copy()
        try:
            window_stc = window_stc.crop(tmin=float(times[0]), tmax=float(times[0]))
        except Exception:
            pass
        window_stc._data = np.asarray(mean_vertex[:, np.newaxis], dtype=float)
        if selected_labels:
            window_stc = self._mask_stc_to_labels(window_stc, selected_labels)

        result = {
            "stimulus_name": preview_label,
            "trial_epoch_index": trial_epoch_index,
            "available_stimulus_groups": available_stimulus_groups,
            "times": times.tolist(),
            "whole_brain_timecourse": whole_brain_tc.tolist(),
            "selected_timecourse": selected_tc.tolist(),
            "window_stc": window_stc,
            "window_start": float(window_start),
            "window_end": float(window_end),
            "available_parcellations": available_parcellations,
            "selected_parcellations": selected_roi_names,
            "subject": str(subject),
            "subjects_dir": str(subjects_dir),
        }
        parcel_msg = f"{len(selected_roi_names)} selected parcellations" if selected_roi_names else "all areas"
        trial_msg = f"single trial {selected_idx[0]}" if len(selected_idx) == 1 and trial_epoch_index is not None else f"{len(selected_idx)} matched trials"
        msg = (
            f"preview stimulus={preview_label}, {trial_msg}, "
            f"window={window_start:.3f}-{window_end:.3f}s, showing {parcel_msg}, method={method}"
        )
        if hasattr(self, "meg_3d") and self.meg_3d is not None and hasattr(self.meg_3d, "display_meg_stc_on_brain"):
            try:
                self.meg_3d.display_meg_stc_on_brain(window_stc, time_sec=float(window_stc.times[0]) if getattr(window_stc, "times", None) is not None and len(window_stc.times) > 0 else None)
            except Exception:
                pass
        return result, msg

    def _compute_semantic_decoding_for_meg_analysis(self, options=None):
        """Decode stimulus groups from source ROI activity in a post-onset window."""
        if options is None:
            options = {}
        if not getattr(self, "b_fif_loaded", False):
            return None, "No FIF loaded."
        epochs = getattr(self, "_fif_epochs", None)
        if epochs is None or len(epochs) == 0:
            return None, "No epochs available for semantic decoding."

        group_mode = str(options.get("group_mode", "canonical") or "canonical").strip().lower()
        if group_mode not in {"canonical", "category", "individual"}:
            group_mode = "canonical"
        try:
            window_start = float(options.get("window_start", 0.10))
            window_end = float(options.get("window_end", 0.30))
        except Exception:
            return None, "Invalid semantic decoding window."
        if not np.isfinite(window_start) or not np.isfinite(window_end) or window_end <= window_start:
            return None, "Semantic decoding requires a valid time window."
        try:
            min_trials = max(2, int(options.get("min_trials_per_class", 2)))
        except Exception:
            min_trials = 2
        method = str(options.get("method", "MNE") or "MNE").strip()
        try:
            lambda2 = float(options.get("lambda2", 1.0 / 9.0))
        except Exception:
            lambda2 = 1.0 / 9.0
        lambda2 = max(lambda2, 1e-12)
        inverse_operator_path = str(options.get("inverse_operator_path", "") or "").strip()
        requested_parcellations = options.get("selected_parcellations", [])
        if not isinstance(requested_parcellations, (list, tuple, np.ndarray)):
            requested_parcellations = []
        requested_parcellations = [str(v) for v in requested_parcellations if str(v).strip()]

        labels_all, trial_df = self._stimulus_labels_for_loaded_epochs(group_mode=group_mode)
        if not labels_all:
            return None, "No stimulus labels available for decoding."
        n = min(len(epochs), len(labels_all))
        epochs = epochs[:n]
        y_all = np.asarray(labels_all[:n], dtype=object)

        valid_mask = np.asarray([bool(str(v).strip()) and str(v).strip().lower() != "unknown" for v in y_all], dtype=bool)
        if not np.any(valid_mask):
            return None, "No valid stimulus labels available for decoding."
        epochs = epochs[valid_mask]
        y_all = y_all[valid_mask]
        trial_df = trial_df.loc[valid_mask].reset_index(drop=True) if not trial_df.empty else pd.DataFrame()

        counts = pd.Series(y_all).value_counts()
        keep_classes = counts[counts >= int(min_trials)].index.tolist()
        if len(keep_classes) < 2:
            return None, "Need at least two stimulus groups with enough trials for decoding."
        keep_mask = np.isin(y_all, keep_classes)
        if not np.any(keep_mask):
            return None, "No trials survived semantic decoding class filtering."
        epochs = epochs[keep_mask]
        y = np.asarray(y_all[keep_mask], dtype=str)
        if not trial_df.empty:
            trial_df = trial_df.loc[keep_mask].reset_index(drop=True)

        class_counts = pd.Series(y).value_counts()
        cv_splits = int(min(5, class_counts.min()))
        if cv_splits < 2:
            return None, "Need at least two trials per class for cross-validated decoding."

        cfg_ok, cfg_msg = self._auto_configure_freesurfer_subject_for_fif(
            getattr(self, "_fif_filename", None),
            require_bem=False,
        )
        if cfg_msg:
            self.c.log.emit(f"MEG semantic decoding config: {cfg_msg}")

        subject = None
        subjects_dir = None
        if hasattr(self, "meg_3d") and self.meg_3d is not None and hasattr(self.meg_3d, "_get_brain_subject_info"):
            try:
                subject, subjects_dir = self.meg_3d._get_brain_subject_info()
            except Exception:
                subject, subjects_dir = None, None
        if not subject or not subjects_dir:
            return None, "No FreeSurfer subject is available for semantic ROI decoding."

        if not inverse_operator_path:
            auto_inv = self._find_inverse_operator_for_fif(getattr(self, "_fif_filename", None))
            if auto_inv and os.path.isfile(auto_inv):
                inverse_operator_path = auto_inv
            else:
                created_path, create_msg = self._create_inverse_operator_for_fif(
                    getattr(self, "_fif_filename", None),
                    output_path=self._default_inverse_operator_output_path(getattr(self, "_fif_filename", None)),
                )
                if create_msg:
                    self.c.log.emit(create_msg)
                if created_path and os.path.isfile(created_path):
                    inverse_operator_path = created_path
        if not inverse_operator_path or not os.path.isfile(inverse_operator_path):
            return None, "Inverse operator is required for semantic decoding."

        available_parcellations = self._get_available_semantic_parcellations(subject=subject, subjects_dir=subjects_dir)
        roi_names = requested_parcellations or available_parcellations
        available_set = set(available_parcellations)
        roi_names = [name for name in roi_names if name in available_set]
        if not roi_names:
            return None, "No semantic ROI definitions available."

        try:
            annot_labels = mne.read_labels_from_annot(
                subject=str(subject),
                parc="aparc",
                subjects_dir=str(subjects_dir),
                verbose="error",
            )
        except Exception as exc:
            return None, f"Could not load aparc labels for semantic decoding: {exc}"

        annot_map = {str(lbl.name): lbl for lbl in annot_labels}
        selected_labels = []
        selected_roi_names = []
        for roi_name in roi_names:
            annot_name = self._ctx_volume_name_to_annot_name(roi_name)
            lbl = annot_map.get(str(annot_name))
            if lbl is not None:
                selected_labels.append(lbl)
                selected_roi_names.append(str(roi_name))
        if not selected_labels:
            return None, "None of the semantic ROI labels were found in the subject annotation."

        try:
            epochs_meg = epochs.copy().pick("meg", exclude="bads")
            if method.upper() == "LCMV":
                stcs = self._compute_lcmv_stcs_from_epochs(epochs_meg)
                inv = None
                src_for_labels = None
                if stcs:
                    src_for_labels = getattr(stcs[0], "src", None)
                if src_for_labels is None:
                    fwd, _noise_cov = self._get_forward_and_noise_for_meg_info(epochs_meg.info)
                    src_for_labels = fwd["src"]
            else:
                inv = mne.minimum_norm.read_inverse_operator(inverse_operator_path, verbose="error")
                stcs = mne.minimum_norm.apply_inverse_epochs(
                    epochs_meg,
                    inv,
                    lambda2=lambda2,
                    method=method,
                    pick_ori=None,
                    return_generator=False,
                    verbose="error",
                )
                src_for_labels = inv["src"]
            if not stcs:
                return None, "No source estimates were produced for semantic decoding."
            times = np.asarray(stcs[0].times, dtype=float)
            window_mask = (times >= window_start) & (times <= window_end)
            if not np.any(window_mask):
                return None, "Semantic window does not overlap the source time axis."
            label_tc = mne.extract_label_time_course(
                stcs,
                selected_labels,
                src_for_labels,
                mode="mean_flip",
                return_generator=False,
                allow_empty=True,
                verbose="error",
            )
        except Exception as exc:
            return None, f"Semantic ROI source extraction failed: {exc}"

        label_tc = np.asarray(label_tc, dtype=float)
        if label_tc.ndim != 3 or label_tc.shape[0] != len(y):
            return None, "Unexpected semantic ROI time-course shape."
        X = np.nanmean(np.abs(label_tc[:, :, window_mask]), axis=2)
        if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] == 0:
            return None, "Semantic decoding features could not be constructed."

        feature_names = [name.replace("ctx-", "") for name in selected_roi_names]
        abs_label_tc = np.abs(label_tc)
        whole_brain_trial_tc = np.asarray(
            [np.nanmean(np.abs(np.asarray(stc.data, dtype=float)), axis=0) for stc in stcs],
            dtype=float,
        )
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced"),
        )
        try:
            y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
            clf.fit(X, y)
        except Exception as exc:
            return None, f"Semantic decoder training failed: {exc}"

        acc = float(accuracy_score(y, y_pred))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        class_names = list(pd.Index(sorted(pd.unique(y))).astype(str))
        cm = confusion_matrix(y, y_pred, labels=class_names)

        report = classification_report(
            y,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
        if "support" in metrics_df.columns:
            metrics_df["support"] = pd.to_numeric(metrics_df["support"], errors="coerce")
        for col in ["precision", "recall", "f1-score"]:
            if col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        lr = clf.named_steps.get("logisticregression")
        coef = getattr(lr, "coef_", None)
        if coef is not None:
            importance = np.mean(np.abs(np.asarray(coef, dtype=float)), axis=0)
        else:
            importance = np.full(X.shape[1], np.nan, dtype=float)
        feature_df = pd.DataFrame(
            {
                "roi": feature_names,
                "importance": importance,
                "mean_signal": np.nanmean(X, axis=0),
                "std_signal": np.nanstd(X, axis=0),
            }
        ).sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)

        trial_feature_df = pd.DataFrame(X, columns=feature_names)
        trial_feature_df.insert(0, "stimulus_group", y)
        if not trial_df.empty:
            if "trial_index" in trial_df.columns:
                trial_feature_df.insert(0, "trial_index", pd.to_numeric(trial_df["trial_index"], errors="coerce").tolist())
            elif "trial" in trial_df.columns:
                trial_feature_df.insert(0, "trial_index", pd.to_numeric(trial_df["trial"], errors="coerce").tolist())
            if "stimulus" in trial_df.columns:
                trial_feature_df.insert(
                    min(2, len(trial_feature_df.columns)),
                    "stimulus",
                    trial_df["stimulus"].astype(str).tolist(),
                )

        group_mean_feature_df = (
            trial_feature_df.groupby("stimulus_group", sort=False)[feature_names]
            .mean()
            .reset_index()
        )
        group_counts = (
            trial_feature_df.groupby("stimulus_group", sort=False)
            .size()
            .rename("n_trials")
            .reset_index()
        )
        group_mean_feature_df = group_mean_feature_df.merge(group_counts, on="stimulus_group", how="left")

        group_timecourse_map = {}
        whole_brain_timecourse_map = {}
        group_window_stc_map = {}
        window_mid = float(np.mean(times[window_mask])) if np.any(window_mask) else float(times[0])
        for class_name in class_names:
            class_mask = np.asarray(y == str(class_name), dtype=bool)
            if np.any(class_mask):
                group_timecourse_map[str(class_name)] = np.nanmean(abs_label_tc[class_mask], axis=0).tolist()
                if whole_brain_trial_tc.ndim == 2 and whole_brain_trial_tc.shape[1] == len(times):
                    whole_brain_timecourse_map[str(class_name)] = np.nanmean(
                        whole_brain_trial_tc[class_mask],
                        axis=0,
                    ).tolist()
                class_indices = np.where(class_mask)[0]
                if class_indices.size > 0:
                    class_window_data = np.stack(
                        [np.abs(np.asarray(stcs[int(idx)].data[:, window_mask], dtype=float)) for idx in class_indices],
                        axis=0,
                    )
                    mean_vertex = np.nanmean(class_window_data, axis=(0, 2))
                    window_stc = stcs[int(class_indices[0])].copy()
                    try:
                        window_stc = window_stc.crop(tmin=float(times[0]), tmax=float(times[0]))
                    except Exception:
                        pass
                    try:
                        window_stc._data = np.asarray(mean_vertex[:, np.newaxis], dtype=float)
                    except Exception:
                        pass
                    group_window_stc_map[str(class_name)] = window_stc

        result = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_trials": int(len(y)),
            "n_classes": int(len(class_names)),
            "class_names": class_names,
            "confusion_matrix": cm,
            "metrics_df": metrics_df,
            "feature_df": feature_df,
            "trial_feature_df": trial_feature_df,
            "group_mean_feature_df": group_mean_feature_df,
            "group_timecourse_map": group_timecourse_map,
            "whole_brain_timecourse_map": whole_brain_timecourse_map,
            "group_window_stc_map": group_window_stc_map,
            "times": times.tolist(),
            "window_start": float(window_start),
            "window_end": float(window_end),
            "available_parcellations": available_parcellations,
            "selected_parcellations": selected_roi_names,
            "roi_names": selected_roi_names,
            "group_mode": group_mode,
            "inverse_operator_path": str(inverse_operator_path),
            "subject": str(subject),
            "subjects_dir": str(subjects_dir),
        }
        msg = (
            f"accuracy={acc:.3f}, balanced_accuracy={bal_acc:.3f}, "
            f"classes={len(class_names)}, trials={len(y)}, "
            f"window={window_start:.3f}-{window_end:.3f}s, rois={len(selected_roi_names)}, "
            f"cv={cv_splits}-fold, method={method}"
        )
        return result, msg

    def _meg_memory_telemetry(self):
        """Return byte estimates for MEG caches not owned by the analysis panel."""
        cache = getattr(self, "_concept_signal_epoch_cache", None)
        total_bytes = 0
        seen_arrays = set()
        if isinstance(cache, dict):
            for payload in cache.values():
                if not isinstance(payload, dict):
                    continue
                for value in payload.values():
                    if not isinstance(value, np.ndarray) or id(value) in seen_arrays:
                        continue
                    seen_arrays.add(id(value))
                    total_bytes += int(value.nbytes)
        parcel_cache = getattr(self, "_parcel_timecourse_cache", None)
        parcel_bytes = 0
        if isinstance(parcel_cache, dict):
            parcel_bytes = sum(
                int(entry.get("bytes", 0) or 0)
                for entry in parcel_cache.values()
                if isinstance(entry, dict)
            )
        return {
            "decoder_cache_bytes": int(total_bytes),
            "decoder_cache_entries": int(len(cache)) if isinstance(cache, dict) else 0,
            "parcel_cache_bytes": int(parcel_bytes),
            "parcel_cache_entries": int(len(parcel_cache)) if isinstance(parcel_cache, dict) else 0,
        }

    def _update_meg_analysis_tabs_from_fif(self):
        """Push FIF pipeline outputs to MEG analysis tabs."""
        if not hasattr(self, "meg_analysis") or self.meg_analysis is None:
            return False
        trial_df = getattr(self, "_fif_trial_response_table", pd.DataFrame())
        report_meta = {}
        if hasattr(self, "stimulus_report") and self.stimulus_report is not None:
            maybe_meta = getattr(self.stimulus_report, "_metadata", {})
            if isinstance(maybe_meta, dict):
                report_meta = dict(maybe_meta)

        source_run_paths = []
        fif_filename = getattr(self, "_fif_filename", None)
        if fif_filename:
            source_run_paths.append(os.path.abspath(str(fif_filename)))
        for path in list(getattr(self, "_fif_topomap_run_paths", []) or []):
            if path:
                path_abs = os.path.abspath(str(path))
                if path_abs not in source_run_paths:
                    source_run_paths.append(path_abs)
        epochs_by_path = getattr(self, "_fif_epochs_by_path", {})
        if isinstance(epochs_by_path, dict):
            for path in epochs_by_path:
                if path:
                    path_abs = os.path.abspath(str(path))
                    if path_abs not in source_run_paths:
                        source_run_paths.append(path_abs)
        if isinstance(trial_df, pd.DataFrame) and "source_fif_path" in trial_df.columns:
            for path in trial_df["source_fif_path"].dropna().tolist():
                if path:
                    path_abs = os.path.abspath(str(path))
                    if path_abs not in source_run_paths:
                        source_run_paths.append(path_abs)

        self.meg_analysis.set_source_compute_callback(self._compute_source_stc_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_parcel_heatmap_callback"):
            self.meg_analysis.set_parcel_heatmap_callback(
                self._compute_parcel_trial_heatmaps_for_meg_analysis
            )
        if hasattr(self.meg_analysis, "set_memory_telemetry_provider"):
            self.meg_analysis.set_memory_telemetry_provider(self._meg_memory_telemetry)
        if hasattr(self.meg_analysis, "set_source_display_callback"):
            self.meg_analysis.set_source_display_callback(self._display_source_stc_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_source_brain_view_callback"):
            self.meg_analysis.set_source_brain_view_callback(self._set_source_brain_view_for_meg_analysis)
        if isinstance(trial_df, pd.DataFrame) and not trial_df.empty:
            try:
                self.meg_analysis.set_trial_data(trial_df, metadata=report_meta)
            except Exception as exc:
                self.c.log.emit(f"MEG analysis trial-table update warning: {exc}")
        try:
            self.meg_analysis.set_mne_context(
                raw=getattr(self, "_fif_raw", None),
                epochs=getattr(self, "_fif_epochs", None),
                evokeds=getattr(self, "_fif_evokeds", {}),
                primary_event=getattr(self, "_fif_primary_event", None),
                primary_evoked=getattr(self, "_fif_primary_evoked", None),
                fif_filename=fif_filename,
                topomap_epochs_by_path=getattr(self, "_fif_epochs_by_path", {}),
                topomap_run_meta_by_path=getattr(self, "_fif_epoch_metadata_by_path", {}),
            )
        except Exception as exc:
            self.c.log.emit(f"MEG analysis MNE-context update warning: {exc}")
        if hasattr(self.meg_analysis, "set_source_run_options"):
            try:
                self.meg_analysis.set_source_run_options(
                    run_paths=source_run_paths,
                    fif_filename=fif_filename,
                    trial_df=trial_df,
                    run_meta_by_path=getattr(self, "_fif_epoch_metadata_by_path", {}),
                )
            except Exception as exc:
                self.c.log.emit(f"MEG source run selector update warning: {exc}")

        self.meg_analysis.set_semantic_preview_callback(self._compute_semantic_preview_for_meg_analysis)
        self.meg_analysis.set_semantic_decode_callback(self._compute_semantic_decoding_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_concept_decode_callback"):
            self.meg_analysis.set_concept_decode_callback(self._compute_concept_decoding_for_meg_analysis)
        inv_path = self._find_inverse_operator_for_fif(getattr(self, "_fif_filename", None))
        subjects_dir = None
        subject = None
        if hasattr(self, "meg_3d") and self.meg_3d is not None and hasattr(self.meg_3d, "_get_brain_subject_info"):
            try:
                subject, subjects_dir = self.meg_3d._get_brain_subject_info()
            except Exception:
                subject = None
                subjects_dir = None
        if hasattr(self.meg_analysis, "set_semantic_parcellations"):
            self.meg_analysis.set_semantic_parcellations(
                self._get_available_semantic_parcellations(subject=subject, subjects_dir=subjects_dir),
                selected=[],
            )
        if hasattr(self.meg_analysis, "set_source_defaults"):
            self.meg_analysis.set_source_defaults(
                inverse_operator_path=inv_path,
                subjects_dir=subjects_dir,
                subject=subject,
            )

        self.meg_analysis.set_source_stc(
            None,
            status_text="No source estimate yet. Use source tab button to compute.",
        )
        return True

    def _update_psd_tab_from_fif(self, max_channels=24, max_seconds=60.0):
        """Compute channel PSDs from loaded FIF and push them to the PSD tab."""
        if not getattr(self, "b_fif_loaded", False):
            return False
        if not hasattr(self, "_fif_raw") or self._fif_raw is None:
            return False
        if not hasattr(self, "power_spectrum") or self.power_spectrum is None:
            return False

        try:
            sfreq = float(getattr(self, "_fif_fs", self._fif_raw.info.get("sfreq", 0.0)))
            if not np.isfinite(sfreq) or sfreq <= 0:
                return False

            selected_names = list(getattr(self, "_fif_channel_names", []))
            if not selected_names:
                return False
            ch_names = selected_names[: max(1, int(max_channels))]
            pick_indices = [self._fif_pick_name_to_idx[ch] for ch in ch_names if ch in self._fif_pick_name_to_idx]
            if not pick_indices:
                return False

            n_total = int(getattr(self, "_fif_total_samples", self._fif_raw.n_times))
            stop = min(n_total, max(1, int(round(float(max_seconds) * sfreq))))
            data_si = self._fif_raw.get_data(
                picks=pick_indices,
                start=0,
                stop=stop,
                reject_by_annotation='omit',
            )
            data_si = np.asarray(data_si, dtype=np.float64)
            if data_si.ndim != 2 or data_si.shape[0] == 0 or data_si.shape[1] < 16:
                return False

            nperseg = int(min(max(256, int(round(2.0 * sfreq))), data_si.shape[1]))
            if nperseg < 16:
                return False
            window = signal.windows.barthann(nperseg)
            eps = 1e-24

            psd_by_channel = {}
            for idx, ch_name in enumerate(ch_names[:data_si.shape[0]]):
                raw_sig = np.asarray(data_si[idx, :], dtype=np.float64)
                if raw_sig.size < nperseg:
                    continue

                f_raw, p_raw = signal.welch(
                    raw_sig,
                    fs=sfreq,
                    window=window,
                    nperseg=nperseg,
                    nfft=nperseg,
                )
                if f_raw is None or p_raw is None or len(f_raw) == 0:
                    continue
                psd_raw_db = 10.0 * np.log10(np.maximum(np.asarray(p_raw, dtype=np.float64), eps))

                psd_filt_db = psd_raw_db
                try:
                    filt_sig = self.filtering.custom_bp_filter(
                        data=raw_sig,
                        fs=sfreq,
                        channel=ch_name,
                    )
                    if filt_sig is not None and len(filt_sig) >= nperseg:
                        f_filt, p_filt = signal.welch(
                            np.asarray(filt_sig, dtype=np.float64),
                            fs=sfreq,
                            window=window,
                            nperseg=nperseg,
                            nfft=nperseg,
                        )
                        if f_filt is not None and p_filt is not None and len(f_filt) > 0:
                            psd_filt_raw_db = 10.0 * np.log10(np.maximum(np.asarray(p_filt, dtype=np.float64), eps))
                            if len(f_filt) == len(f_raw) and np.allclose(f_filt, f_raw):
                                psd_filt_db = psd_filt_raw_db
                            else:
                                psd_filt_db = np.interp(
                                    f_raw,
                                    np.asarray(f_filt, dtype=np.float64),
                                    psd_filt_raw_db,
                                    left=float(psd_filt_raw_db[0]),
                                    right=float(psd_filt_raw_db[-1]),
                                )
                except Exception:
                    psd_filt_db = psd_raw_db

                psd_by_channel[str(ch_name)] = {
                    "freq": np.asarray(f_raw, dtype=np.float64),
                    "psd_raw": np.asarray(psd_raw_db, dtype=np.float64),
                    "psd_filtered": np.asarray(psd_filt_db, dtype=np.float64),
                }

            if not psd_by_channel:
                return False

            loaded = self.power_spectrum.load_precomputed_psd_data(
                psd_by_channel=psd_by_channel,
                fs=sfreq,
                source_label=f"FIF ({os.path.basename(str(getattr(self, '_fif_filename', 'data')) )})",
                clear_notches=True,
            )
            return bool(loaded)
        except Exception:
            print(traceback.format_exc())
            return False

    def _update_mne_psd_tab_from_fif(self, max_channels=None, max_seconds=60.0):
        """Render MNE tutorial-style PSD plot in the PSD tab for loaded FIF."""
        if not getattr(self, "b_fif_loaded", False):
            return False
        if not hasattr(self, "power_spectrum") or self.power_spectrum is None:
            return False
        if not hasattr(self, "_fif_raw") or self._fif_raw is None:
            return False
        if not hasattr(self.power_spectrum, "plot_mne_spectrum"):
            return False

        try:
            spectrum = getattr(self, "_fif_psd", None)
            if spectrum is None:
                raw = self._fif_raw
                sfreq = float(raw.info["sfreq"])
                stop = min(int(raw.n_times), max(1, int(round(float(max_seconds) * sfreq))))
                meg_picks, bad_meg_names = good_meg_picks(raw.info, meg=True, ref_meg=False)
                if bad_meg_names:
                    emit_bad_meg_channel_log(
                        self.c.log,
                        "FIF PSD",
                        fif_path=getattr(self, "_fif_filename", None),
                        bad_names=bad_meg_names,
                    )
                if meg_picks.size == 0:
                    return False
                if max_channels is None:
                    pick_subset = meg_picks
                else:
                    pick_subset = meg_picks[: max(1, int(max_channels))]
                raw_small = raw.copy().pick(pick_subset)
                raw_small.crop(tmin=0.0, tmax=max(0.0, (stop - 1) / sfreq))
                spectrum = raw_small.compute_psd(
                    fmin=1.0,
                    fmax=120.0,
                    tmin=0.0,
                    tmax=min(float(raw_small.times[-1]), float(max_seconds)),
                    verbose='error',
                )
                self._fif_psd = spectrum

            source_label = f"FIF ({os.path.basename(str(getattr(self, '_fif_filename', 'data')))})"
            return bool(self.power_spectrum.plot_mne_spectrum(spectrum, source_label=source_label))
        except Exception:
            print(traceback.format_exc())
            if hasattr(self.power_spectrum, "clear_mne_psd_plot"):
                self.power_spectrum.clear_mne_psd_plot(
                    message="Could not compute/render MNE PSD for this FIF file."
                )
            return False
            
    #region Read NSx file (Ripple and Blackrock)
    
    def read_ripple_nsx_file(self, filename, analog_only=False, segment_only=False, 
                             event_only=False, skip=0, 
                             max_segment=100, max_analog=90000):
        """
        Reads a Ripple NSx file and stores the entities in a list.

        Args:
            filename: path to the NSx file
            analog_only: if True, only analog entities will be stored
            segment_only: if True, only segment entities will be stored
            event_only: if True, only event entities will be stored
            skip: number of entities to skip
            max_segment: maximum number of segments to read
            max_analog: maximum number of analog entities to read
        """
        try:
            self.b_fif_loaded = False
            self._reset_fif_analysis_state()
            self._reset_fif_analysis_outputs_in_3d()
            
            self.study_info = StudyInfo(nsx_file_path=filename, c=self.c)
            self.ripple_nsx.legacy = self.study_info.study_loaded
            if hasattr(self.study_info, 'notches_info'):
                self.filtering.notches_info = self.study_info.notches_info
            if len(self.study_info.image_names) > 0:
                self.responses_list.load_study_info(study_info=self.study_info)
                # self.responses_list.rasters = self.study_info.rasters
                self.responses_list.rasters = self.study_info.grapes.get('rasters', None)
            
            
            self.ripple_nsx.read_nsx_file(filename, analog_only=analog_only, segment_only=segment_only,
                                        event_only=event_only, skip=skip, max_segment=max_segment, 
                                        max_analog=max_analog)
        except:
            print(traceback.format_exc())
            self.c.log.emit(f'Error reading Ripple NSx file {filename}')

        return self.study_info.study_loaded
    
    def post_read_nsx(self, state):
        if self.ripple_nsx.b_nsx_read:
            self.b_nsx_read = True
            channel_dict = self.ripple_nsx.ripple_map.channel_dict
            self.signal_viewer.update_channel_list(channel_dict)

    def post_clustering_nsx(self, state):
        """
        Updates the cluster info.
        """
        if state:
            if self.ripple_nsx.legacy:
                self.clustering.update_clustering_cmb_channels(ch_lbls=self.ripple_nsx.micro_ch_labels)
                self.clustering.load_clus_info_dict_from_times_mat(study_folder=self.study_info.study_folder, 
                                                                   nsx_info=self.study_info.nsx)
                self.responses_list.load_response_metrics_threaded(clus_info_dict=self.clustering.clus_info_dict)
            else:
                self.clustering.update_clustering_cmb_channels(ch_lbls=self.ripple_nsx.micro_ch_labels)
                # Start computing response metrics like ifr, z_score, etc.
                self.responses_list.clus_info_dict = self.clustering.clus_info_dict
                self.responses_list.compute_response_metrics_mu_threaded(clus_info_dict=self.clustering.clus_info_dict,
                    micro_ch_lbls=self.ripple_nsx.micro_ch_labels)

    def post_spk_det_online(self, daq_events=None, photo_times=None):
        """
        Updates the responses/grapes view.
        """
        if self.clustering.seq_id == 0 and self.clustering.subscr_id == 0:
            self.clustering.update_clustering_cmb_channels(self.ripple_acq.analysis_chs_lbls)           
            self.responses_list.update_responses_cmb_channels(self.ripple_acq.analysis_chs_lbls)
        
        self.clustering.remove_all_collisions(ch_lbls=self.ripple_acq.analysis_chs_lbls)
        self.responses_list.clus_info_dict = self.clustering.clus_info_dict
        b_last_subscr_seq = self.clustering.seq_id + 2 == self.clustering.subscr_seq_count
        self.responses_list.compute_response_metrics_mu(micro_ch_lbls=self.ripple_acq.analysis_chs_lbls,
                                                    daq_events=daq_events,
                                                    photo_times=photo_times,
                                                    last_subscr_seq=b_last_subscr_seq)
        # For debugging
        # self.responses_list.compute_response_metrics_mu(micro_ch_lbls=self.ripple_acq.analysis_chs_lbls,
        #                                             daq_events=daq_events,
        #                                             photo_times=photo_times)
        if b_last_subscr_seq:
            self.do_sorting()
        elif self.clustering.seq_id > 1:
            self.do_sorting(template_matching=True)

    def do_sorting(self, template_matching=False):
        """
        Starts the online clustering process.
        """
        if template_matching:
            self.clustering.do_template_matching_all_channels(ch_lbls=self.ripple_acq.analysis_chs_lbls)
        else:
            self.clustering.do_spike_sorting(ch_lbls=self.ripple_acq.analysis_chs_lbls)

    def post_clustering_online(self):
        """
        Updates the responses/grapes view.
        """ 
        self.clustering.plot_clusters(channel=self.ripple_acq.analysis_chs_lbls[0])
        self.responses_list.clus_info_dict = self.clustering.clus_info_dict
        self.responses_list.compute_response_metrics_su_threaded(micro_ch_lbls=self.ripple_acq.analysis_chs_lbls)             













    def read_brk_ns_file(self, filename):
        self.b_fif_loaded = False
        self._reset_fif_analysis_state()
        self._reset_fif_analysis_outputs_in_3d()
        self.brk_nsx = BlackrockNsx()
        self.brk_nsx.read_brk_ns_file(filename)


    def load_fif_file_for_3d(self, filename, report_fif_files=None):
        """Load FIF into 3D viewer and SignalViewer."""
        loaded_3d = False
        loaded_signals = False
        analysis_filename = self._resolve_ica_cleaned_fif(filename)
        analysis_report_fif_files = self._resolve_ica_cleaned_fif_files(report_fif_files or [])

        if hasattr(self, 'meg_3d') and self.meg_3d is not None:
            if hasattr(self.meg_3d, "_register_loaded_meg_fif"):
                try:
                    for path in [analysis_filename] + list(analysis_report_fif_files or []):
                        self.meg_3d._register_loaded_meg_fif(path)
                except Exception:
                    pass
            try:
                cfg_ok, cfg_msg = self._auto_configure_freesurfer_subject_for_fif(
                    analysis_filename,
                    require_bem=False,
                )
                if cfg_msg:
                    self.c.log.emit(f"MEG viewer model: {cfg_msg}")
                if not cfg_ok:
                    self.c.log.emit("MEG viewer model: falling back to fsaverage.")
            except Exception as exc:
                self.c.log.emit(f"MEG viewer model setup warning: {exc}")
            loaded_3d = bool(self.meg_3d.load_fif_sensors(analysis_filename))
            if loaded_3d and hasattr(self.meg_3d, "set_meg_source_auto"):
                try:
                    # Keep FIF loading lightweight; source projection is explicit.
                    self.meg_3d.set_meg_source_auto(False, project_now=False)
                except Exception:
                    pass

        loaded_signals = self.load_fif_signals(
            analysis_filename,
            report_fif_files=analysis_report_fif_files,
        )
        if loaded_signals and hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, "set_meg_source_auto"):
            try:
                self.meg_3d.set_meg_source_auto(False, project_now=False)
            except Exception:
                pass
        return bool(loaded_3d or loaded_signals)

    def load_fif_signals(self, filename, report_fif_files=None):
        """Load MEG signals from FIF for windowed display in SignalViewer."""
        try:
            self._reset_fif_analysis_state()
            filename = os.path.abspath(str(filename))
            raw = mne.io.read_raw_fif(filename, preload=False, verbose='error')
            info = raw.info

            mag_picks, bad_mag_names = good_meg_picks(info, meg='mag', ref_meg=False)
            grad_picks, bad_grad_names = good_meg_picks(info, meg='grad', ref_meg=False)
            if bad_mag_names:
                emit_bad_meg_channel_log(
                    self.c.log,
                    "FIF signal display (magnetometers)",
                    fif_path=filename,
                    bad_names=bad_mag_names,
                )
            if bad_grad_names:
                emit_bad_meg_channel_log(
                    self.c.log,
                    "FIF signal display (gradiometers)",
                    fif_path=filename,
                    bad_names=bad_grad_names,
                )
            if mag_picks.size == 0 and grad_picks.size == 0:
                self.c.log.emit("No MEG magnetometer or gradiometer channels found in FIF file.")
                return False

            # Keep the primary analysis selection unchanged (magnetometers when
            # available), while exposing both sensor types in the signal viewer.
            picks = mag_picks if mag_picks.size else grad_picks
            scale = 1e15 if mag_picks.size else 1e13
            unit_label = "fT" if mag_picks.size else "fT/cm"
            display_picks = [int(p) for p in mag_picks] + [int(p) for p in grad_picks]
            self._fif_raw = raw
            self._fif_picks = [int(p) for p in picks]
            self._fif_channel_names = [raw.ch_names[i] for i in self._fif_picks]
            self._fif_magnetometer_names = [raw.ch_names[int(i)] for i in mag_picks]
            self._fif_gradiometer_names = [raw.ch_names[int(i)] for i in grad_picks]
            self._fif_display_channel_names = [raw.ch_names[i] for i in display_picks]
            self._fif_pick_name_to_idx = {raw.ch_names[i]: int(i) for i in display_picks}
            self._fif_channel_scale_by_name = {
                **{name: 1e15 for name in self._fif_magnetometer_names},
                **{name: 1e13 for name in self._fif_gradiometer_names},
            }
            self._fif_channel_unit_by_name = {
                **{name: "fT" for name in self._fif_magnetometer_names},
                **{name: "fT/cm" for name in self._fif_gradiometer_names},
            }
            self._fif_scale = float(scale)
            self._fif_unit_label = unit_label
            self._fif_start_sample = 0
            self._fif_total_samples = int(raw.n_times)
            self._fif_fs = float(info['sfreq'])
            self._fif_filename = str(filename)
            self._fif_ica_metadata = self._read_ica_metadata(filename)
            self._fif_original_filename = (
                self._fif_ica_metadata.get("source_fif_path")
                if isinstance(self._fif_ica_metadata, dict) and self._fif_ica_metadata.get("source_fif_path")
                else str(filename)
            )

            self.b_fif_loaded = True
            self.signal_viewer.fs = int(round(self._fif_fs))
            self.signal_viewer.ph_diode = False
            self.signal_viewer.pics = False

            # Keep filter settings valid for MEG sampling rates.
            self.filtering.fs = self.signal_viewer.fs
            nyq = max(1.0, self._fif_fs / 2.0)
            if self.filtering.highcut >= nyq:
                self.filtering.highcut = min(120.0, nyq * 0.9)
            if self.filtering.lowcut >= self.filtering.highcut:
                self.filtering.lowcut = max(0.1, self.filtering.highcut / 4.0)

            max_visible = max(1, int(self.signal_viewer.num_chs_to_plot) - 2)
            self._fif_selected_channels = list(self._fif_channel_names[:max_visible])
            self._fif_events_selected = True
            channel_dict = self._build_fif_channel_tree(
                selected_channels=set(self._fif_selected_channels)
            )
            self.signal_viewer.update_channel_list(channel_dict)
            self.update_fif_signal_window()
            pipeline_ok = False
            try:
                pipeline_ok = self._compute_fif_mne_pipeline()
            except Exception as exc:
                self.c.log.emit(f'FIF MNE pipeline error: {exc}')
                print(traceback.format_exc())

            if pipeline_ok and isinstance(self._fif_analysis_summary, dict):
                summary = self._fif_analysis_summary
                cov_method = summary.get('noise_cov_selected_method') or summary.get('noise_cov_method') or 'none'
                self.c.log.emit(
                    "FIF MNE pipeline ready: "
                    f"events={summary.get('n_events', 0)} ({summary.get('event_strategy', 'n/a')}), "
                    f"epochs={summary.get('n_epochs', 0)}, "
                    f"conditions={summary.get('n_conditions', 0)}, "
                    f"primary={summary.get('primary_event', 'n/a')}, "
                    f"psd_ch={summary.get('n_psd_channels', 0)}, "
                    f"tfr_ch={summary.get('n_tfr_channels', 0)}, "
                    f"noise_cov={summary.get('noise_cov_kind', 'none')}/{cov_method}, "
                    f"source={'yes' if summary.get('source_projected', False) else 'no'}"
                )
                # Refresh to draw event markers now that pipeline data is available.
                self.update_fif_signal_window()
            else:
                self.c.log.emit("FIF MNE pipeline partial: using raw/window playback only.")

            psd_loaded = False
            try:
                psd_loaded = bool(self._update_psd_tab_from_fif())
            except Exception as exc:
                self.c.log.emit(f'FIF PSD tab update error: {exc}')
                print(traceback.format_exc())
            if psd_loaded:
                self.c.log.emit("Loaded FIF PSD into PSD tab.")
            else:
                self.c.log.emit("Could not populate PSD tab from FIF data.")

            mne_psd_loaded = False
            try:
                mne_psd_loaded = bool(self._update_mne_psd_tab_from_fif())
            except Exception as exc:
                self.c.log.emit(f'FIF MNE PSD plot error: {exc}')
                print(traceback.format_exc())
            if mne_psd_loaded:
                self.c.log.emit("Loaded MNE-style PSD plot in PSD tab.")
            else:
                self.c.log.emit("Could not render MNE-style PSD plot.")

            stim_report_loaded = False
            try:
                stim_report_loaded = bool(
                    self._compute_stimulus_report_from_event_logs(
                        filename,
                        report_fif_files=report_fif_files,
                    )
                )
            except Exception as exc:
                self.c.log.emit(f'FIF stimulus report error: {exc}')
                print(traceback.format_exc())
            if stim_report_loaded:
                n_rows = int(len(getattr(self, "_fif_trial_response_table", pd.DataFrame())))
                n_fif = 1
                if getattr(self, "stimulus_report", None) is not None:
                    report_meta = getattr(self.stimulus_report, "_metadata", {})
                    if isinstance(report_meta, dict):
                        try:
                            n_fif = int(report_meta.get("n_fif_processed", 1))
                        except Exception:
                            n_fif = 1
                self.c.log.emit(
                    f"Loaded stimulus response report from events_log ({n_rows} mapped trials, {n_fif} FIF file(s))."
                )
            else:
                self.c.log.emit("Could not generate stimulus response report from events_log.")

            try:
                self._prepare_topomap_epoch_runs([filename] + list(report_fif_files or []))
            except Exception as exc:
                self.c.log.emit(f"Could not prepare run-selectable topomap epochs: {exc}")
                print(traceback.format_exc())

            try:
                self._update_meg_analysis_tabs_from_fif()
            except Exception as exc:
                self.c.log.emit(f"Could not populate MEG analysis tabs: {exc}")
                print(traceback.format_exc())

            self.c.log.emit(
                "Loaded FIF signals: "
                f"{len(self._fif_magnetometer_names)} magnetometers, "
                f"{len(self._fif_gradiometer_names)} gradiometers, "
                f"fs={self._fif_fs:.1f} Hz"
            )
            return True
        except Exception as exc:
            self.c.log.emit(f"Error loading FIF signals: {exc}")
            print(traceback.format_exc())
            return False

    def _build_fif_channel_tree(self, selected_channels=None):
        """Build a tree dictionary compatible with CheckableTreeComboBox."""
        if selected_channels is None:
            selected_channels = set()
        tree = {
            FIF_EVENTS_CHANNEL_LABEL: (
                bool(getattr(self, "_fif_events_selected", True)),
                {},
            ),
        }
        groups = (
            ("Magnetometers", getattr(self, "_fif_magnetometer_names", [])),
            ("Gradiometers", getattr(self, "_fif_gradiometer_names", [])),
        )
        for group_label, channel_names in groups:
            if not channel_names:
                continue
            children = {ch: (ch in selected_channels, {}) for ch in channel_names}
            tree[group_label] = (any(ch in selected_channels for ch in channel_names), children)
        return tree

    def _extract_checked_leaf_channels(self, channel_dict):
        """Extract checked leaf channel labels from a channel tree dict."""
        selected = []

        def walk(node):
            for label, value in node.items():
                if not isinstance(value, tuple) or len(value) < 2:
                    continue
                check_state, children = value
                is_checked = int(check_state) != 0
                if isinstance(children, dict) and children:
                    walk(children)
                elif is_checked:
                    selected.append(label)

        if isinstance(channel_dict, dict):
            walk(channel_dict)
        return selected

    def _update_fif_selected_channels_from_tree(self, channel_dict):
        """Apply checked channel selection from UI to FIF playback."""
        selected = self._extract_checked_leaf_channels(channel_dict)
        self._fif_events_selected = FIF_EVENTS_CHANNEL_LABEL in selected
        valid = [ch for ch in selected if ch in self._fif_pick_name_to_idx]
        if not valid:
            valid = list(self._fif_channel_names[:max(1, self.signal_viewer.num_chs_to_plot - 2)])
        self._fif_selected_channels = valid
        self._limit_fif_selected_channels()

    def _limit_fif_selected_channels(self):
        """Limit selected FIF channels to visible waveform rows."""
        if not getattr(self, "b_fif_loaded", False):
            return
        max_visible = max(1, int(self.signal_viewer.num_chs_to_plot) - 2)
        if not getattr(self, "_fif_selected_channels", None):
            self._fif_selected_channels = list(self._fif_channel_names[:max_visible])
        self._fif_selected_channels = self._fif_selected_channels[:max_visible]

    def update_fif_signal_window(self):
        """Push current FIF window into SignalViewer."""
        if not getattr(self, "b_fif_loaded", False):
            return

        self._limit_fif_selected_channels()
        if not self._fif_selected_channels:
            return

        fs = float(self._fif_fs)
        n_total = int(self._fif_total_samples)
        window_samples = max(1, int(round(float(self.signal_viewer.length) * fs)))
        max_start = max(0, n_total - window_samples)
        self._fif_start_sample = int(np.clip(self._fif_start_sample, 0, max_start))
        start = self._fif_start_sample
        stop = min(n_total, start + window_samples)

        pick_indices = [self._fif_pick_name_to_idx[ch] for ch in self._fif_selected_channels]
        data_si = self._fif_raw.get_data(
            picks=pick_indices,
            start=start,
            stop=stop,
            reject_by_annotation='omit',
        )
        data_si = np.asarray(data_si, dtype=np.float64)
        scales = np.asarray(
            [self._fif_channel_scale_by_name[ch] for ch in self._fif_selected_channels],
            dtype=np.float64,
        )[:, np.newaxis]
        data = np.asarray(data_si * scales, dtype=np.float32)
        if data.shape[1] < window_samples:
            pad = np.zeros((data.shape[0], window_samples - data.shape[1]), dtype=np.float32)
            data = np.hstack((data, pad))
            data_si = np.hstack(
                (data_si, np.zeros((data_si.shape[0], window_samples - data_si.shape[1]), dtype=np.float64))
            )

        t = np.arange(window_samples, dtype=np.float32) / fs
        window_events = self._fif_window_events(start_sample=start, stop_sample=stop)
        display_data = data
        display_channels = list(self._fif_selected_channels)
        if bool(getattr(self, "_fif_events_selected", True)):
            event_trace = np.zeros(window_samples, dtype=np.float32)
            events = getattr(self, "_fif_events", None)
            if events is not None and len(events):
                events = np.asarray(events)
                mask = (events[:, 0] >= start) & (events[:, 0] < stop)
                for sample, _, code in events[mask]:
                    event_trace[int(sample) - start] = float(code)
            display_data = np.vstack((event_trace, data))
            display_channels.insert(0, FIF_EVENTS_CHANNEL_LABEL)
        vis_buffer = {
            'data': display_data,
            't': t,
            'ts': int(start),
            'events': window_events,
            'disp_chs': display_channels,
            'vis_buffer_size': int(window_samples),
            'vis_sample_idx': int(start),
        }
        self.signal_viewer.update_vis_buffer(vis_buffer)

        # Cache window data for optional MNE inverse projection (expects SI units).
        if hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, 'set_meg_projection_window'):
            try:
                self.meg_3d.set_meg_projection_window(
                    data_si,
                    list(self._fif_selected_channels),
                    fs,
                    start_sample=int(start),
                    tmin_sec=0.0,
                )
            except Exception:
                pass

        # Update 3D sensor colors on the current brain surface based on window activity.
        if hasattr(self, 'meg_3d') and self.meg_3d is not None and hasattr(self.meg_3d, 'set_meg_sensor_activity'):
            valid_n = max(1, int(stop - start))
            activity_scores = {}
            for idx, ch_name in enumerate(self._fif_selected_channels):
                ch_data = data[idx, :valid_n]
                if ch_data.size == 0:
                    continue
                activity_scores[ch_name] = float(np.sqrt(np.mean(np.square(ch_data))))
            self.meg_3d.set_meg_sensor_activity(
                activity_scores,
                unit_label=" / ".join(dict.fromkeys(
                    self._fif_channel_unit_by_name[ch] for ch in self._fif_selected_channels
                )),
            )

    def navigate_fif(self, direction='forward'):
        """Move FIF playback by one window length."""
        if not getattr(self, "b_fif_loaded", False):
            return
        step = max(1, int(round(float(self.signal_viewer.length) * float(self._fif_fs))))
        if direction == 'forward':
            self._fif_start_sample += step
        else:
            self._fif_start_sample -= step
        self.update_fif_signal_window()

    def init_power_spectrum_layout(self):
        """
        Initializes the power spectrum plots.
        """
        self.power_spectrum = PowerSpectrum(psd_view=self.psd_view, c=self.c, filtering=self.filtering)



    #region Stimulus report

    def init_stimulus_report_layout(self):
        self.meg_analysis = MEGAnalysisPanel(report_view=self.stim_report_view, c=self.c)
        # Keep legacy name for existing report-generation code paths.
        self.stimulus_report = self.meg_analysis.stimulus_report
        self.meg_analysis.set_source_compute_callback(self._compute_source_stc_for_meg_analysis)
        self.meg_analysis.set_semantic_preview_callback(self._compute_semantic_preview_for_meg_analysis)
        self.meg_analysis.set_semantic_decode_callback(self._compute_semantic_decoding_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_concept_decode_callback"):
            self.meg_analysis.set_concept_decode_callback(self._compute_concept_decoding_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_parcel_heatmap_callback"):
            self.meg_analysis.set_parcel_heatmap_callback(
                self._compute_parcel_trial_heatmaps_for_meg_analysis
            )
        if hasattr(self.meg_analysis, "set_memory_telemetry_provider"):
            self.meg_analysis.set_memory_telemetry_provider(self._meg_memory_telemetry)
        if hasattr(self.meg_analysis, "set_source_display_callback"):
            self.meg_analysis.set_source_display_callback(self._display_source_stc_for_meg_analysis)
        if hasattr(self.meg_analysis, "set_source_brain_view_callback"):
            self.meg_analysis.set_source_brain_view_callback(self._set_source_brain_view_for_meg_analysis)

    #endregion Stimulus report

    #region 3D viewer
        
    def init_3d_layout(self):
        self.meg_3d = MEG3DViewer(chs_3d_view=self.meg_3d_view, c=self.c)
        
    #endregion 3D viewer

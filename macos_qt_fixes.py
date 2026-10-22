"""
macOS Qt Compatibility Fixes
============================

This module contains fixes and workarounds for Qt compatibility issues on macOS,
specifically targeting the QMacCGContext paint engine warnings and PyQtGraph
overflow issues.
"""

import os
import sys
import warnings
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

def apply_macos_qt_fixes():
    """Apply various Qt fixes and settings for better macOS compatibility."""
    
    # Environment variables to suppress Qt warnings
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.cocoa.drawing.debug=false")
    os.environ.setdefault("QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    
    # Prevent multiple dock icons by setting the application to run as a single process
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("PYQT_IGNORE_GPU_BLACKLIST", "1")
    
    # Configure multiprocessing to not create new dock icons
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    
    # Python-specific environment variables for child processes
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")  # Prevent .pyc files
    os.environ.setdefault("PYTHONUNBUFFERED", "1")  # Unbuffered output
    
    # Additional Qt logging suppressions
    qt_suppression_rules = [
        "qt.qpa.cocoa.drawing.debug=false",
        "qt.qpa.cocoa.window.debug=false",
        "qt.qpa.cocoa.view.debug=false",
        "qt.qpa.cocoa.graphics.debug=false"
    ]
    os.environ["QT_LOGGING_RULES"] = ";".join(qt_suppression_rules)

def configure_app_for_macos(app):
    """Configure QApplication settings for better macOS compatibility."""
    
    if not isinstance(app, QtWidgets.QApplication):
        return
    
    # Set application attributes for macOS
    app.setAttribute(QtCore.Qt.AA_DisableWindowContextHelpButton, True)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    
    # macOS specific attributes
    if hasattr(QtCore.Qt, 'AA_DontShowIconsInMenus'):
        app.setAttribute(QtCore.Qt.AA_DontShowIconsInMenus, False)
    
    if hasattr(QtCore.Qt, 'AA_MacDontSwapCtrlAndMeta'):
        app.setAttribute(QtCore.Qt.AA_MacDontSwapCtrlAndMeta, False)
    
    # Prevent multiple Python icons in dock
    if hasattr(QtCore.Qt, 'AA_DontUseNativeMenuBar'):
        app.setAttribute(QtCore.Qt.AA_DontUseNativeMenuBar, False)
    
    # Configure application to appear as a single unified app
    app.setQuitOnLastWindowClosed(True)
    
    # Set application organization and name for proper macOS integration
    app.setOrganizationName("Wired Brains Lab")
    app.setOrganizationDomain("mcw.edu")
    app.setApplicationName("MAS")
    app.setApplicationDisplayName("MAS")
    app.setApplicationVersion("1.0")
    
    # Set style for better compatibility
    try:
        # Try to use the native macOS style
        available_styles = QtWidgets.QStyleFactory.keys()
        if 'macOS' in available_styles:
            app.setStyle('macOS')
        elif 'Fusion' in available_styles:
            app.setStyle('Fusion')
    except Exception as e:
        print(f"Could not set Qt style: {e}")
    
    # Configure dock icon behavior
    try:
        # Try to set the activation policy to regular app
        import objc
        from Foundation import NSBundle
        from AppKit import NSApplication, NSApplicationActivationPolicyRegular
        
        # Set macOS bundle identity
        set_macos_app_identity("MAS")
        
        # Get the current application
        ns_app = NSApplication.sharedApplication()
        
        # Set activation policy to regular (shows in dock as single app)
        ns_app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        
        print("macOS dock integration configured successfully")
    except ImportError:
        print("PyObjC not available, using Qt-only configuration")
    except Exception as e:
        print(f"Could not configure macOS dock integration: {e}")

def set_macos_app_identity(app_name="MAS"):
    """Set macOS application name in NSBundle and NSProcessInfo.
    
    This MUST be called before QApplication is created so that macOS
    initializes the menu bar (the application menu next to the Apple logo)
    with 'MAS' rather than 'python' or 'MNE-Python'.
    """
    if sys.platform != "darwin":
        return
        
    try:
        from Foundation import NSBundle, NSProcessInfo
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = app_name
            info["CFBundleDisplayName"] = app_name
        
        proc_info = NSProcessInfo.processInfo()
        if proc_info is not None and hasattr(proc_info, "setProcessName_"):
            proc_info.setProcessName_(app_name)
    except Exception as e:
        print(f"Could not set macOS bundle identity: {e}")

def protect_app_name(app_name="MAS"):
    """Prevent external libraries (like MNE-Python) from overwriting the application name."""
    try:
        _orig_core_set = QtCore.QCoreApplication.setApplicationName
        
        def _safe_set_app_name(*args, **kwargs):
            actual_args = [a for a in args if not isinstance(a, QtCore.QCoreApplication)]
            name = actual_args[0] if actual_args else kwargs.get('application')
            if name in ('MNE-Python', 'python', 'Python'):
                return
            return _orig_core_set(*actual_args, **kwargs)
        
        QtCore.QCoreApplication.setApplicationName = _safe_set_app_name
        if hasattr(QtWidgets.QApplication, 'setApplicationName'):
            QtWidgets.QApplication.setApplicationName = _safe_set_app_name
    except Exception as e:
        print(f"Could not protect app name: {e}")

def patch_mne_qtapp(app_name="MAS", org_name="Wired Brains Lab", org_domain="mcw.edu"):
    """Patch MNE's Qt application initialization to prevent it from renaming MAS."""
    try:
        import mne.viz.backends._utils as mne_utils
        orig_init = getattr(mne_utils, '_init_mne_qtapp', None)
        if orig_init is not None and not getattr(orig_init, '_mas_patched', False):
            def wrapped_init(*args, **kwargs):
                kwargs['enable_icon'] = False
                res = orig_init(*args, **kwargs)
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    app.setApplicationName(app_name)
                    app.setApplicationDisplayName(app_name)
                    app.setOrganizationName(org_name)
                    app.setOrganizationDomain(org_domain)
                return res
            wrapped_init._mas_patched = True
            mne_utils._init_mne_qtapp = wrapped_init
            
            for mod_name in ['mne.viz.backends._pyvista', 'mne.viz.backends._qt', 'mne.utils.check']:
                if mod_name in sys.modules:
                    setattr(sys.modules[mod_name], '_init_mne_qtapp', wrapped_init)
    except Exception:
        pass

def create_qt_message_handler():
    """Create a custom Qt message handler to filter out unwanted warnings."""
    
    def qt_message_handler(mode, context, message):
        # List of message patterns to suppress
        suppressed_patterns = [
            "QMacCGContext:: Unsupported paint engine type",
            "QPainter::begin: Paint device returned engine",
            "QOpenGLShaderProgram::uniformLocation",
            "QCocoaWindow::setVisible",
            "QCocoaBackingStore::flush",
            "CGContextSaveGState",
            "CGContextRestoreGState",
            "overflow encountered in cast",
            "ViewBox",
            "RuntimeWarning"
        ]
        
        # Check if message should be suppressed
        for pattern in suppressed_patterns:
            if pattern in message:
                return
        
        # Print non-suppressed messages with appropriate prefix
        prefix_map = {
            QtCore.QtMsgType.QtDebugMsg: "Qt Debug",
            QtCore.QtMsgType.QtWarningMsg: "Qt Warning", 
            QtCore.QtMsgType.QtCriticalMsg: "Qt Critical",
            QtCore.QtMsgType.QtFatalMsg: "Qt Fatal",
            QtCore.QtMsgType.QtInfoMsg: "Qt Info"
        }
        
        prefix = prefix_map.get(mode, "Qt Message")
        print(f"[{prefix}]: {message}")
    
    return qt_message_handler

def configure_pyqtgraph_fixes():
    """Configure PyQtGraph to handle numerical overflow issues."""
    try:
        import pyqtgraph as pg
        
        # Configure PyQtGraph to handle large numbers better
        pg.setConfigOption('crashWarning', True)  # Show crash warnings
        pg.setConfigOption('enableExperimental', False)  # Disable experimental features that might cause issues
        
        # Set up numpy error handling for overflow warnings
        np.seterr(over='ignore', invalid='ignore')
        
        # Suppress specific PyQtGraph-related warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*overflow encountered in cast.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='.*ViewBox.*')
        
        print("PyQtGraph numerical fixes applied")
        
    except ImportError:
        print("PyQtGraph not available, skipping numerical fixes")
    except Exception as e:
        print(f"Could not apply PyQtGraph fixes: {e}")

def suppress_child_process_dock_icon():
    """Call this function in child processes to suppress dock icon on macOS."""
    if sys.platform != "darwin":
        return
        
    try:
        # Try to set the process as an agent (no dock icon)
        import objc
        from AppKit import NSApplication, NSApplicationActivationPolicyProhibited
        
        # Get or create the application instance
        app = NSApplication.sharedApplication()
        
        # Set activation policy to prohibited (no dock icon)
        app.setActivationPolicy_(NSApplicationActivationPolicyProhibited)
        
        print("Child process dock icon suppressed")
    except ImportError:
        print("PyObjC not available in child process")
    except Exception as e:
        print(f"Could not suppress child process dock icon: {e}")



def setup_macos_multiprocessing_wrapper():
    """Setup environment for multiprocessing on macOS - removed wrapper due to pickle issues."""
    if sys.platform != "darwin":
        return
    
    # Just ensure the spawn method is set - the individual process functions
    # now handle dock icon suppression themselves
    import multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("macOS multiprocessing spawn method configured")

def setup_macos_compatibility():
    """Setup all macOS compatibility fixes. Call this early in main()."""
    
    # Set macOS bundle identity early (before QApplication creation)
    set_macos_app_identity("MAS")
    
    # Protect application name against external renames
    protect_app_name("MAS")
    
    # Patch MNE Qt initialization to prevent overriding app name
    patch_mne_qtapp("MAS")

    # Apply environment variable fixes first
    apply_macos_qt_fixes()
    
    # Configure PyQtGraph fixes
    configure_pyqtgraph_fixes()
    
    # Install multiprocessing wrapper for macOS
    setup_macos_multiprocessing_wrapper()
    
    # Install custom message handler
    message_handler = create_qt_message_handler()
    QtCore.qInstallMessageHandler(message_handler)
    
    print("macOS Qt compatibility fixes applied")

# Apply bundle identity immediately at import time on macOS
if sys.platform == "darwin":
    set_macos_app_identity("MAS")
    protect_app_name("MAS")
    patch_mne_qtapp("MAS")

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
from the_retry import retry

from .enums import Mode, Scale
from .exceptions import ElementNotFound, NotFile
from .logging import Level, log

if sys.platform == "win32":
    import clipboard
    import win32api
    import win32con
    from pywinauto import ElementNotFoundError, timings
    from pywinauto.application import Application, ProcessNotFoundError
    from pywinauto.keyboard import send_keys
    from pywinauto.timings import TimeoutError
else:
    Application = Any  # type: ignore[misc,assignment]

    # Placeholders so module-level decorators evaluate on non-Windows platforms.
    # The Gigapixel constructor raises OSError before any of them are used.
    class ElementNotFoundError(Exception):  # type: ignore[no-redef]
        pass

    class ProcessNotFoundError(Exception):  # type: ignore[no-redef]
        pass

    timings = None  # type: ignore[assignment]


class Gigapixel:
    """
    Windows-only automation wrapper for Topaz Gigapixel AI.
    """

    def __init__(
        self,
        executable_path: Union[Path, str],
        processing_timeout: int = 900,
    ) -> None:
        self._executable_path = Path(executable_path)
        self._processing_timeout = processing_timeout

        if sys.platform != "win32":
            raise OSError("Gigapixel automation is only supported on Windows.")

        instance = self._get_instance()
        self._app = self._App(instance, processing_timeout)

    class _App:
        def __init__(self, app: Application, processing_timeout: int):
            timings.Timings.window_find_timeout = 0.5
            self.app = app
            self._processing_timeout = processing_timeout
            self._main_window = self.app.window()

            self.scale: Optional[Scale] = None
            self.mode: Optional[Mode] = None

            self._cancel_btn: Optional[Any] = None
            self._save_btn: Optional[Any] = None
            self._scale_buttons: Dict[Scale, Any] = {}
            self._mode_buttons: Dict[Mode, Any] = {}

        @retry(
            expected_exception=(ElementNotFoundError,),
            attempts=5,
            backoff=0.5,
            exponential_backoff=True,
        )
        @log("Opening photo: {}", "Photo opened", format=(1,), level=Level.DEBUG)
        def open_photo(self, photo_path: Path) -> None:
            while photo_path.name not in self._main_window.element_info.name:
                logger.debug("Trying to open photo")
                self._main_window.set_focus()
                send_keys("{ESC}^o")
                clipboard.copy(str(photo_path))
                send_keys("^v {ENTER}{ESC}")

        @log("Saving photo", "Photo saved", level=Level.DEBUG)
        def save_photo(self) -> None:
            self._open_export_dialog()
            send_keys("{ENTER}")
            if self._cancel_btn is None:
                self._cancel_btn = self._main_window.child_window(
                    title="Close window", control_type="Button", depth=1
                )
            self._cancel_btn.wait("visible", timeout=self._processing_timeout)
            self._close_export_dialog()

        @retry(
            expected_exception=(TimeoutError,),
            attempts=10,
            backoff=0.1,
            exponential_backoff=True,
        )
        @log("Opening export dialog", "Export dialog opened", level=Level.DEBUG)
        def _open_export_dialog(self) -> None:
            send_keys("^S")
            if self._save_btn is None:
                self._save_btn = self._main_window.child_window(
                    title="Save", control_type="Button", depth=1
                )
            self._save_btn.wait("visible", timeout=0.1)

        @retry(
            expected_exception=(TimeoutError,),
            attempts=10,
            backoff=0.1,
            exponential_backoff=True,
        )
        @log("Closing export dialog", "Export dialog closed", level=Level.DEBUG)
        def _close_export_dialog(self) -> None:
            send_keys("{ESC}")
            self._cancel_btn.wait_not("visible", timeout=0.1)

        @log("Setting processing options", "Processing options set", level=Level.DEBUG)
        def set_processing_options(
            self,
            scale: Optional[Scale] = None,
            mode: Optional[Mode] = None,
        ) -> None:
            if scale:
                self._set_scale(scale)
            if mode:
                self._set_mode(mode)

        def _set_scale(self, scale: Scale) -> None:
            if self.scale == scale:
                return
            try:
                if scale not in self._scale_buttons:
                    self._scale_buttons[scale] = self._main_window.child_window(
                        title=scale.value
                    )
                self._scale_buttons[scale].click_input()
            except ElementNotFoundError as exc:
                raise ElementNotFound(f"Scale button {scale.value} not found") from exc
            self.scale = scale
            logger.debug(f"Scale set to {scale.value}")

        def _set_mode(self, mode: Mode) -> None:
            if self.mode == mode:
                return
            try:
                if mode not in self._mode_buttons:
                    self._mode_buttons[mode] = self._main_window.child_window(
                        title=mode.value
                    )
                self._mode_buttons[mode].click_input()
            except ElementNotFoundError as exc:
                raise ElementNotFound(f"Mode button {mode.value} not found") from exc
            self.mode = mode
            logger.debug(f"Mode set to {mode.value}")

    @log(start="Getting Gigapixel instance...")
    @log(end="Got Gigapixel instance: {}", format=(-1,), level=Level.SUCCESS)
    def _get_instance(self) -> Application:
        try:
            return Application(backend="uia").connect(path=str(self._executable_path))
        except ProcessNotFoundError:
            logger.debug("Gigapixel not running; starting new instance.")
            return self._open_topaz()

    @log("Starting new Gigapixel instance...", "Started new Gigapixel instance: {}",
         format=(-1,), level=Level.DEBUG)
    def _open_topaz(self) -> Application:
        return Application(backend="uia").start(
            str(self._executable_path)
        ).connect(path=str(self._executable_path))

    @log("Checking path: {}", "Path is valid", format=(1,), level=Level.DEBUG)
    def _check_path(self, path: Path) -> None:
        if not path.is_file():
            raise NotFile(f"Path is not a file: {path}")

    @staticmethod
    def _set_english_layout() -> None:
        english_layout = 0x0409
        win32api.LoadKeyboardLayout(hex(english_layout), win32con.KLF_ACTIVATE)

    @log(start="Starting processing: {}", format=(1,))
    @log(end="Finished processing: {}", format=(1,), level=Level.SUCCESS)
    def process(
        self,
        photo_path: Union[Path, str],
        scale: Optional[Scale] = None,
        mode: Optional[Mode] = None,
    ) -> None:
        photo_path = Path(photo_path)
        self._set_english_layout()
        self._check_path(photo_path)

        self._app.open_photo(photo_path)
        self._app.set_processing_options(scale, mode)
        self._app.save_photo()

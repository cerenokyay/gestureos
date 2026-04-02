import time
from dataclasses import dataclass, field


@dataclass
class Cooldown:
    cooldown_sec: float = 0.7
    last: dict = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.time()
        if now - self.last.get(key, 0.0) < self.cooldown_sec:
            return False
        self.last[key] = now
        return True


import pyautogui


class KeyPress:
    def __init__(self, key: str):
        self.key = key

    def run(self, mouse_pos=None) -> None:
        pyautogui.press(self.key)


class Hotkey:
    def __init__(self, keys: list[str]):
        self.keys = keys

    def run(self, mouse_pos=None) -> None:
        pyautogui.hotkey(*self.keys)


class Message:
    def __init__(self, text: str):
        self.text = text

    def run(self, mouse_pos=None) -> None:
        print(f"💬 {self.text}")


class MouseMove:
    """Mouse'u belirtilen koordinata hareket ettir"""
    def __init__(self):
        pass

    def run(self, mouse_pos=None) -> None:
        if mouse_pos:
            pyautogui.moveTo(mouse_pos[0], mouse_pos[1], duration=0)


class MouseClick:
    """Mouse tıklaması (sol click)"""
    def __init__(self):
        pass

    def run(self, mouse_pos=None) -> None:
        pyautogui.click()

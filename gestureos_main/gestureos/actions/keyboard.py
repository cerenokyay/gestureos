import pyautogui

class KeyPress:
    def __init__(self, key: str):
        self.key = key

    def run(self) -> None:
        pyautogui.press(self.key)

class Hotkey:
    def __init__(self, keys: list[str]):
        self.keys = keys

    def run(self) -> None:
        pyautogui.hotkey(*self.keys)
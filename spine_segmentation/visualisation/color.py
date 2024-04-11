from typing import Tuple, TypeVar, Union

import numpy as np


class Color:
    colors = {
        "grey": (128, 128, 128, 255),
        "red": (255, 0, 0, 255),
        "lightred": (255, 144, 144, 255),
        "green": (0, 255, 0, 255),
        "lightgreen": (144, 238, 144, 255),
        "darkgreen": (0, 100, 0, 255),
        "blue": (0, 0, 255, 255),
        "yellow": (255, 255, 0, 255),
        "orange": (255, 165, 0, 255),
        "brown": (165, 42, 42, 255),
        "cyan": (0, 255, 255, 255),
        "magenta": (255, 0, 255, 255),
        "violet": (238, 130, 238, 255),
        "purple": (128, 0, 128, 255),
        "black": (0, 0, 0, 255),
        "white": (255, 255, 255, 255),
        "transparent": (255, 255, 255, 0),
    }
    ANSI_RESET = "\033[0m"
    ANSI_BOLD = "\033[1m"

    def __init__(
        self,
        red_or_grey_or_color: "ColorLike",
        green: Union[float, int, None] = None,
        blue: Union[float, int, None] = None,
        /,
        alpha: Union[float, int, None] = None,
    ):
        """Color either as 1 (grey), 3 (RGB), 4 (RGBA) numbers, a string ("red") or hexcode ("#ff0000")

        Can be in range of either [0.0, 1.0] if floats or [0, 255] if ints. Must be all same type.
        If alpha not supplied, it is implied to be 255 (fully opaque).
        If first argument is str, the other arguments are ignored.
        """
        if isinstance(red_or_grey_or_color, Color):
            red, green, blue, alpha = red_or_grey_or_color.ints()
        elif isinstance(red_or_grey_or_color, tuple):
            red, green, blue, alpha = Color(*red_or_grey_or_color).ints()
        elif isinstance(red_or_grey_or_color, str):
            red, green, blue, alpha = self._from_str(red_or_grey_or_color)
        else:
            red_or_grey = red_or_grey_or_color
            if green is None and blue is None:
                green = blue = red_or_grey
            elif green is None or blue is None:
                raise TypeError(f"One of green={green} or blue={blue} is supplied, but not both!")
            red = red_or_grey
            if alpha is None:
                alpha = 1.0 if isinstance(red, float) else 255
        if not (type(red) is type(green) is type(blue) is type(alpha)):
            raise TypeError(
                f"Color variables not of the same type: "
                f"{type(red)} != {type(green)} != {type(blue)} != {type(alpha)}"
            )
        self._color: Tuple[int, int, int, int] = red, green, blue, alpha
        if isinstance(red, float):
            self._color = tuple(map(lambda c: int(c * 255), self._color))
        assert len(self._color) == 4
        if not all(0 <= c <= 255 for c in self._color):
            raise ValueError(f"Color out of bounds: {self._color} (must be between 0 and 255)")

    def ints(self) -> Tuple[int, int, int, int]:
        return self._color

    def floats(self) -> Tuple[float, float, float, float]:
        return tuple(c / 255 for c in self._color)

    def hex(self) -> str:
        return "#" + "".join(f"{c:02x}" for c in self._color)

    def inverted(self) -> "Color":
        return Color(255 - self.r, 255 - self.g, 255 - self.b, self.a)

    def ansi(self) -> str:
        return f"\033[38;2;{';'.join(map(str, self._color))}m"

    def ansi_surround(self, text: str, bold: bool = False) -> str:
        additional = ""
        if bold:
            additional += self.ANSI_BOLD
        return additional + self.ansi() + text + self.ANSI_RESET

    def hsv(self) -> Tuple[float, float, float, float]:
        # Based on conversion from https://en.wikipedia.org/wiki/HSL_and_HSV
        r, g, b, alpha = self.floats()
        color_max, color_min = max(r, g, b), min(r, g, b)
        delta = color_max - color_min
        if delta == 0:
            hue = 0
        elif color_max == r:
            hue = ((g - b) / delta) % 6
        elif color_max == g:
            hue = ((b - r) / delta) + 2
        elif color_max == b:
            hue = ((r - g) / delta) + 4
        hue = 60.0 * hue
        # angle_alpha = 0.5 * (2 * r - g - b)
        # angle_beta = np.sqrt(3) * 2 * (g - b)
        # hue_angle = np.atan2(angle_beta, angle_alpha)
        value = color_max
        saturation = 0 if value == 0 else delta / value
        return hue, saturation, value, alpha

    @staticmethod
    def from_hsv(hue: float, saturation: float, value: float, alpha: float = 1.0) -> "Color":
        """Create a Color object from a hsv value

        @param hue: angle on the color wheel hue [0.0, 360.0)
        @param saturation: percentage of saturation [0.0, 1.0]
        @param value: percentage for brightness [0.0, 1.0]
        @param alpha: percentage opaqueness [0.0, 1.0] (0 is fully transparent, 1 is fully opaque)
        @return: Color object
        """
        hue %= 360
        c = value * saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = value - c
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(hue // 60)]
        r, g, b = map(float, tuple((np.array(rgb) + m)))
        return Color(r, g, b, alpha)

    @staticmethod
    def _calculate_alpha_interpolation_factors(arr, interpolate_from, take_new_factor):
        alpha = 3
        with np.errstate(divide="ignore", invalid="ignore"):
            new_factors = (
                take_new_factor * interpolate_from[..., alpha] / (arr[..., alpha] + interpolate_from[..., alpha])
            )
        np.nan_to_num(new_factors, copy=False, nan=1)
        new_factors = new_factors[..., np.newaxis]
        return new_factors, 1 - new_factors

    @staticmethod
    def make_gradient(color1, color2, number_of_values):
        # which takes 2 numbers as tuples of 4 floats and creates a list of number of values which are slowly interpolated from color1 to color2
        # color1 and color2 are in rgb format
        take2_factor = np.arange(number_of_values)[:, None] / number_of_values
        take1_factor = 1 - take2_factor
        gradient = color1.floats() * take1_factor + color2.floats() * take2_factor
        return [Color(*c) for c in gradient]

    @staticmethod
    def fast_interpolate_rgb_arrays(arr, interpolate_from, take_new_factor=0.6):
        """
        Expected shape for arrays: [x, y, 4], where the last dimension is red, green, blue, alpha.
        Hue is in degrees from 0 to 360, saturation, value and alpha are each between 0 and 1
        @param arr:
        @param interpolate_from:
        @param take_new_factor:
        @return:
        """
        assert arr.shape[-1] == 4 == interpolate_from.shape[-1], "Arrays do not contain color data"
        new_factors, old_factors = Color._calculate_alpha_interpolation_factors(arr, interpolate_from, take_new_factor)
        arr = arr * old_factors + interpolate_from * new_factors
        return arr

    @staticmethod
    def fast_interpolate_hsv_arrays(arr, interpolate_from, take_new_factor=0.6):
        """
        Expected shape for arrays: [x, y, 4], where the last dimension is red, green, blue, alpha.
        Hue is in degrees from 0 to 360, saturation, value and alpha are each between 0 and 1
        @param arr:
        @param interpolate_from:
        @param take_new_factor:
        @return:
        """
        assert arr.shape[-1] == 4 == interpolate_from.shape[-1], "Arrays do not contain color data"
        hue, saturation, value, alpha = 0, 1, 2, 3
        new_factors, old_factors = Color._calculate_alpha_interpolation_factors(arr, interpolate_from, take_new_factor)
        # Find the shortest angle between the 2 angles
        # proof: https://math.stackexchange.com/questions/2144234/interpolating-between-2-angles
        # The 540 was replaced with 180, because in python modulo of negative numbers returns positive numbers,
        # e.g.: -400 % 360 = 320 (instead of -40, like in many other programming languages). So a 540 is unnecessary
        # here, as it as used to guarantee that the number is positive
        angles_between = ((interpolate_from[..., hue] - arr[..., hue]) % 360 + 180) % 360 - 180
        arr[..., hue] = (arr[..., hue] + angles_between * new_factors[..., 0]) % 360
        sat_alpha = slice(saturation, alpha + 1)
        arr[..., sat_alpha] = arr[..., sat_alpha] * old_factors + interpolate_from[..., sat_alpha] * new_factors
        return arr

    @staticmethod
    def hsv_array_to_rgb_array(hsv_array: np.ndarray) -> np.ndarray:
        assert hsv_array.shape[-1] == 4, "Array does not contain color data in last axis (or does not contain alpha)"
        hue, value, saturation, alpha = (hsv_array[..., i] for i in range(4))
        hue %= 360
        chroma = value * saturation
        x = chroma * (1 - abs((hue / 60) % 2 - 1))
        m = value - chroma
        hue_as_index = (hue // 60).astype(np.uint8)
        hue_as_index = hue_as_index[np.newaxis, ...]
        zeros = np.zeros_like(hue)
        red = np.take_along_axis(np.stack([chroma, x, zeros, zeros, x, chroma]), hue_as_index, axis=0)[0]
        green = np.take_along_axis(np.stack([x, chroma, chroma, x, zeros, zeros]), hue_as_index, axis=0)[0]
        blue = np.take_along_axis(np.stack([zeros, zeros, x, chroma, chroma, x]), hue_as_index, axis=0)[0]
        rgba_stacked = np.stack([red, green, blue, alpha], axis=-1)
        mmm_no_alpha = np.stack([m, m, m, np.zeros_like(m)], axis=-1)
        # Add m value to rgb, but not to alpha
        rgba_array = ((rgba_stacked + mmm_no_alpha) * 255).round().astype(np.uint8)
        return rgba_array

    @staticmethod
    def _from_hex(color_hex: str) -> Tuple[int, int, int, int]:
        """Create color from hex code such as #af22e3 (RGB) or #3545bf5f (RGBA), # is optional"""
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = color_hex[0] * 2 + color_hex[1] * 2 + color_hex[2] * 2
        if len(color_hex) == 6:
            color_hex += "FF"
        import re

        if not re.fullmatch(r"#?[0-9a-fA-F]{8}", color_hex):
            raise ValueError(f"Invalid or unknown color hex code: {color_hex}!")
        assert len(color_hex) == 8
        colors = tuple(int(color_hex[i : i + 2], 16) for i in range(0, len(color_hex), 2))
        return colors

    @staticmethod
    def _from_str(color_str: str) -> Tuple[int, int, int, int]:
        # Assume transparency, similar to https://en.wikibooks.org/wiki/LaTeX/Colors
        # <COLOR>!transparency-percent (<COLOR>!100==<COLOR>, <COLOR>!0=="transparent"
        if "!" in color_str:
            color, percent = color_str.split("!", maxsplit=1)
            color_int = Color._from_str(color)
            try:
                return *color_int[:3], int(color_int[3] * int(percent) / 100)
            except ValueError:
                pass  # Will raise error at end of function
        try:
            return Color._from_hex(color_str)
        except ValueError:
            pass  # Will raise error at end of function
        if color_str.lower() in Color.colors:
            return Color.colors[color_str.lower()]
        raise ValueError(f"Unknown color_string: {color_str}")

    def __eq__(self, other: "Color"):
        return self._color == other.ints()

    def __str__(self):
        return "Color=" + str(self._color)

    def __repr__(self):
        return "Color=" + str(dict(zip("rgba", self._color)))

    @property
    def r(self) -> int:
        return self._color[0]

    @property
    def g(self) -> int:
        return self._color[1]

    @property
    def b(self) -> int:
        return self._color[2]

    @property
    def a(self) -> int:
        return self._color[3]

    @property
    def visible(self):
        return self.a > 0

    @classmethod
    def random(cls):
        from random import random

        return Color(random(), random(), random())


ColorLike = TypeVar(
    "ColorLike",
    int,
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    float,
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    Color,
    str,  # hex-code or string
)

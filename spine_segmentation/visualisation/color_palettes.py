from collections import OrderedDict
from functools import lru_cache
from typing import Dict

import seaborn as sns

from spine_segmentation.visualisation.color import Color


@lru_cache
def get_alternating_palette() -> Dict[int, Color]:
    vertebra_count = 27
    disc_count = 26

    # vertebra_palette = sns.cubehelix_palette(n_colors=26, start=0.5, rot=-0.75)
    # disc_palette = sns.color_palette("viridis", n_colors=25)
    vertebra_palette = sns.color_palette("Paired", n_colors=vertebra_count)
    disc_palette = sns.color_palette("Paired", n_colors=disc_count + 1)[:-1]
    # vertebra_palette = Color.make_gradient(Color("orange"), Color("red"), vertebra_count)
    # disc_palette = Color.make_gradient(Color("green"), Color("cyan"), disc_count)
    vertebra_palette = [Color(*c) for c in vertebra_palette]
    disc_palette = [Color(*c) for c in disc_palette]
    disc_palette = disc_palette[2:] + disc_palette[:2]

    vertebra_colors = dict(zip(range(1, 100, 2), vertebra_palette))
    disc_colors = dict(zip(range(2, 100, 2), disc_palette))
    # Swap blue and light blue in order to remove S1 (light blue) being adjacent to light blue (48)
    disc_colors[46] = disc_colors[48]
    disc_colors[48] = disc_colors[52]
    palette = {0: Color("black"), **vertebra_colors, **disc_colors}
    ordered = OrderedDict(sorted(palette.items()))
    for i in set(palette.keys()) - set(range(1, 50)):
        del ordered[i]
    return ordered


def get_wk_bs_palette() -> Dict[int, Color]:
    return OrderedDict(
        {
            0: Color("black"),
            1: Color("orange"),
            2: Color(
                0.0,
                0.5,
                0.9,
            ),
        }
    )

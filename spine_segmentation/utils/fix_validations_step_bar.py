import sys

from pytorch_lightning.callbacks import TQDMProgressBar


class FixValidationStepBar(TQDMProgressBar):
    # https://github.com/Lightning-AI/lightning/issues/15283#issuecomment-1289654353
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

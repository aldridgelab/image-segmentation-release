# ``run_particle_labeler`` is a PyQt5-backed UI tool defined in
# ``_annotate.py``. Importing it eagerly would force a PyQt5 dependency on
# every consumer of ``momia2``, including this pipeline's headless feature
# extraction. We expose a wrapper that defers the actual PyQt5 import until
# the labeler is invoked — headless callers can ``import momia2`` freely;
# only an actual call to ``run_particle_labeler`` will need PyQt5.


def run_particle_labeler(*args, **kwargs):
    from ._annotate import run_particle_labeler as _impl
    return _impl(*args, **kwargs)


__all__ = ["run_particle_labeler"]

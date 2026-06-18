from diffusion.consistency.tsp import TSPConsistency

try:
    from diffusion.consistency.mis import MISConsistency
    _MIS_IMPORT_ERROR = None
except Exception as exc:
    _MIS_IMPORT_ERROR = exc

    class MISConsistency:
        def __init__(self, *args, **kwargs):
            raise ImportError("MIS consistency requires torch_sparse") from _MIS_IMPORT_ERROR

from ..logging import logger


def load_dummy_extension():
    try:
        import agoge_dummy_cuda
        import torch

        a = torch.ones(5, device="cuda")
        b = torch.ones(5, device="cuda")
        c = torch.zeros(5, device="cuda")

        agoge_dummy_cuda.dummy_add(a, b, c)
        logger.info(f"CUDA Dummy Add Success. Output: {c}")
    except ImportError:
        logger.warning("agoge_dummy_cuda not built. Run `make cuda-build` first.")
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to run CUDA extension: {e}")

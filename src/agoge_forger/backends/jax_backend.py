from ..logging import logger

def check_jax_env():
    try:
        import jax
        logger.info(f"JAX Version: {jax.__version__}")
        devices = jax.devices()
        logger.info(f"JAX Devices: {devices}")
        logger.info("JAX is currently stubbed for future algorithmic research.")
    except ImportError:
        logger.warning("JAX is not installed. Use 'pip install -e .[jax]' to enable.")

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def default_message(epoch: int, loss: float, time_per_epoch: float):
    return (
        f"Epoch {epoch}: "
        f"Loss = {loss:.3f}, "
        f"Time per epoch = {time_per_epoch:.3f} ms"
    )

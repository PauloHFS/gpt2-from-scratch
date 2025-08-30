class MetricsTracker:

    train_loss: list[float]
    val_loss: list[float]
    throughput: list[float]
    gpu_memory: list[float]

    def __init__(self) -> None:
        self.train_loss = list()
        self.val_loss = list()
        self.throughput = list()
        self.gpu_memory = list()

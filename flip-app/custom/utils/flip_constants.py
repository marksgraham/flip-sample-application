from enum import Enum
import os


class FlipConstants(object):
    CLEANUP: str = "cleanup"
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT")
    MIN_CLIENTS: int = 2


class FlipEvents(object):
    DATA_RETRIEVAL_STARTED = "_data_retrieval_started"
    RESULTS_UPLOAD_STARTED = "_results_upload_started"
    RESULTS_UPLOAD_COMPLETED = "_results_upload_completed"
    RESULTS_UPLOAD_ERROR = "_results_upload_error"
    CLEANUP_ERROR = "_cleanup_error"
    ABORTED = "_aborted"


class ModelStatus(str, Enum):
    PENDING = "PENDING"
    INITIATED = "INITIATED"
    PREPARED = "PREPARED"
    TRAINING_STARTED = "TRAINING_STARTED"
    RESULTS_UPLOADED = "RESULTS_UPLOADED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"

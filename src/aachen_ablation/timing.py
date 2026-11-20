"""Phase timings with one CUDA completion wait, rather than per-image waits."""
import logging
import time

import torch


class PhaseTimer:
    def __init__(self, device):
        self.device = device
        self.started = time.perf_counter()
        self.data_loading_seconds = 0.
        self.host_compute_seconds = 0.
        self.events, self.decoder_events = [], []
        self.decoder_host_seconds = 0.
        self.gpu_timing_available = device.type == "cuda"
        self.last = self.started

    def event(self):
        if not self.gpu_timing_available:
            return None
        try:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device.index))
            return event
        except Exception as exc:
            self.gpu_timing_available = False
            logging.getLogger(__name__).warning("Optional CUDA event timing unavailable: %s", exc)
            return None

    def loaded(self):
        now = time.perf_counter()
        self.data_loading_seconds += now - self.last
        self.compute_started = now
        self.begin = self.event()

    def decoder_done(self):
        self.decoder_host_seconds += time.perf_counter() - self.compute_started
        end = self.event()
        if end is not None:
            self.decoder_events.append((self.begin, end))

    def computed(self):
        self.host_compute_seconds += time.perf_counter() - self.compute_started
        end = self.event()
        if end is not None:
            self.events.append((self.begin, end))
        self.last = time.perf_counter()

    def finish(self):
        gpu_seconds = None
        decoder_seconds = self.decoder_host_seconds if self.device.type == "cpu" else None
        if self.device.type == "cuda":
            # Completion/correctness errors are real device failures, not telemetry.
            torch.cuda.synchronize(self.device.index)
            if self.gpu_timing_available:
                try:
                    gpu_seconds = sum(start.elapsed_time(end) for start, end in self.events)/1000
                    decoder_seconds = sum(start.elapsed_time(end) for start, end in self.decoder_events)/1000
                except Exception as exc:
                    logging.getLogger(__name__).warning("Optional CUDA event timing unavailable: %s", exc)
                    gpu_seconds = decoder_seconds = None
        return {"decoder_seconds": decoder_seconds, "phase_seconds": time.perf_counter()-self.started,
                "data_loading_seconds": self.data_loading_seconds,
                "host_compute_seconds": self.host_compute_seconds,
                "gpu_computation_seconds": gpu_seconds}

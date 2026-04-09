"""OpenTelemetry tracing setup.

Activates only if opentelemetry-sdk is installed. If not, all calls
become no-ops so we never crash a deployment that doesn't have it.
"""
import logging
import os

logger = logging.getLogger("inferbox")

_tracer = None
_initialized = False


def init_tracing(service_name: str = "inferbox"):
    """Initialise OpenTelemetry tracing.

    Reads OTEL_EXPORTER_OTLP_ENDPOINT to determine where to send traces.
    Falls back to console exporter if not set.
    """
    global _tracer, _initialized
    if _initialized:
        return _tracer
    _initialized = True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"OpenTelemetry exporting to {otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not installed, using console")
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            # Console exporter for local debugging
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        logger.info("OpenTelemetry tracing initialised")
        return _tracer
    except ImportError:
        logger.info("opentelemetry-sdk not installed, tracing disabled")
        return None


def get_tracer():
    return _tracer


class _NoopSpan:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def set_attribute(self, *a, **kw):
        pass


def span(name: str, **attrs):
    """Context manager that creates a span if tracing is enabled."""
    if _tracer is None:
        return _NoopSpan()
    sp = _tracer.start_as_current_span(name)
    return sp

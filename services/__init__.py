# Package initialization file
import logging

# Configure a null handler to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

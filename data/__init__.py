# This file marks the directory as a Python package
import logging

# Configure a null handler to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

import os
import sys

# Add the indexer directory to sys.path so build_corpus can be imported
# regardless of which directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(__file__))

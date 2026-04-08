import sys
import os

# Server folder path ni Python ki chepthunnam
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

# Ikkada nundi server folder lo unna original code ni pilusthundhi
from server.inference import *

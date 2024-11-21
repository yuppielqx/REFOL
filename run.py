import config
import sys
from config import config
sys.path.append('./fl-server')
from refol import REFOL


if __name__ == "__main__":

    fl_server = {
        "refol": REFOL(config)  # REFOL
    }[config.agg_model]
    fl_server.boot()
    fl_server.run()
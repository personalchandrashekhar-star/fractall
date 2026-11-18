"""Episode files -> LeRobot dataset. Run it with the LeRobot python (nwaves-dataset does):

    /opt/lerobot-venv/bin/python scripts/to_lerobot.py EPISODE_DIR OUT_DIR --repo-id user/so101_pickplace

The work is done by nwaves_hackathon/to_lerobot.py; this file only makes it runnable from a clone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nwaves_hackathon.to_lerobot import main  # noqa: E402

if __name__ == "__main__":
    main()

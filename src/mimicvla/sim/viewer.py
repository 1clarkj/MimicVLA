"""Run the scripted USB task in MuJoCo's native interactive viewer."""

import threading
import time

import glfw
import mujoco
import mujoco.viewer

from .demo import scripted_actions
from .env import PandaUSBEnv


def run_viewer() -> None:
    """Open a free-camera MuJoCo window and repeat the 20 Hz demo path."""
    env = PandaUSBEnv()
    env.model.vis.scale.framelength = 0.25
    env.model.vis.scale.framewidth = 0.03
    paused = threading.Event()
    restart = threading.Event()
    cycle_frames = threading.Event()
    frame_modes = (
        (mujoco.mjtFrame.mjFRAME_SITE, mujoco.mjtLabel.mjLABEL_SITE, "site frames"),
        (mujoco.mjtFrame.mjFRAME_WORLD, mujoco.mjtLabel.mjLABEL_NONE, "world frame"),
        (mujoco.mjtFrame.mjFRAME_NONE, mujoco.mjtLabel.mjLABEL_NONE, "frames hidden"),
    )
    frame_mode_index = 0

    def update_frame_mode(viewer: mujoco.viewer.Handle) -> None:
        nonlocal frame_mode_index
        if not cycle_frames.is_set():
            return
        cycle_frames.clear()
        frame_mode_index = (frame_mode_index + 1) % len(frame_modes)
        with viewer.lock():
            viewer.opt.frame, viewer.opt.label, name = frame_modes[frame_mode_index]
        print(f"Showing {name}.", flush=True)

    def on_key(key: int) -> None:
        if key == glfw.KEY_SPACE:
            if paused.is_set():
                paused.clear()
            else:
                paused.set()
        elif key == glfw.KEY_R:
            restart.set()
        elif key == glfw.KEY_F:
            cycle_frames.set()

    print("MuJoCo viewer: drag to orbit, scroll to zoom, Space to pause, R to reset, "
          "F to cycle site/world/hidden frames. RGB axes: X red, Y green, Z blue.", flush=True)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = env.config.socket.position_m
                viewer.cam.distance = 0.7
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                viewer.opt.frame, viewer.opt.label, _ = frame_modes[frame_mode_index]
            while viewer.is_running():
                update_frame_mode(viewer)
                with viewer.lock():
                    env.reset()
                restart.clear()
                viewer.sync()
                for action in scripted_actions(env):
                    if not viewer.is_running() or restart.is_set():
                        break
                    while paused.is_set() and viewer.is_running() and not restart.is_set():
                        update_frame_mode(viewer)
                        viewer.sync()
                        time.sleep(0.05)
                    if not viewer.is_running() or restart.is_set():
                        break
                    started = time.monotonic()
                    update_frame_mode(viewer)
                    with viewer.lock():
                        env.step(action)
                    viewer.sync()
                    time.sleep(max(0.0, env.control_dt - (time.monotonic() - started)))
                if viewer.is_running() and not restart.is_set():
                    time.sleep(2.0)
    finally:
        env.close()

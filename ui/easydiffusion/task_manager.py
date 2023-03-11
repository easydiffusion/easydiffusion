"""task_manager.py: manage tasks dispatching and render threads.
Notes:
    render_threads should be the only hard reference held by the manager to the threads.
    Use weak_thread_data to store all other data using weak keys.
    This will allow for garbage collection after the thread dies.
"""
import json
import traceback

TASK_TTL = 15 * 60  # seconds, Discard last session's task timeout

import torch
import queue, threading, time, weakref
from typing import Any, Hashable

from easydiffusion import device_manager
from easydiffusion.types import TaskData, GenerateImageRequest
from easydiffusion.utils import log

from sdkit.utils import gc

THREAD_NAME_PREFIX = ""
ERR_LOCK_FAILED = " failed to acquire lock within timeout."
LOCK_TIMEOUT = 15  # Maximum locking time in seconds before failing a task.
# It's better to get an exception than a deadlock... ALWAYS use timeout in critical paths.

DEVICE_START_TIMEOUT = 60  # seconds - Maximum time to wait for a render device to init.


class SymbolClass(type):  # Print nicely formatted Symbol names.
    def __repr__(self):
        return self.__qualname__

    def __str__(self):
        return self.__name__


class Symbol(metaclass=SymbolClass):
    pass


class ServerStates:
    class Init(Symbol):
        pass

    class LoadingModel(Symbol):
        pass

    class Online(Symbol):
        pass

    class Rendering(Symbol):
        pass

    class Unavailable(Symbol):
        pass


class RenderTask:  # Task with output queue and completion lock.
    def __init__(self, req: GenerateImageRequest, task_data: TaskData):
        task_data.request_id = id(self)
        self.render_request: GenerateImageRequest = req  # Initial Request
        self.task_data: TaskData = task_data
        self.response: Any = None  # Copy of the last reponse
        self.render_device = None  # Select the task affinity. (Not used to change active devices).
        self.temp_images: list = [None] * req.num_outputs * (1 if task_data.show_only_filtered_image else 2)
        self.error: Exception = None
        self.lock: threading.Lock = threading.Lock()  # Locks at task start and unlocks when task is completed
        self.buffer_queue: queue.Queue = queue.Queue()  # Queue of JSON string segments

    async def read_buffer_generator(self):
        try:
            while not self.buffer_queue.empty():
                res = self.buffer_queue.get(block=False)
                self.buffer_queue.task_done()
                yield res
        except queue.Empty as e:
            yield

    @property
    def status(self):
        if self.lock.locked():
            return "running"
        if isinstance(self.error, StopAsyncIteration):
            return "stopped"
        if self.error:
            return "error"
        if not self.buffer_queue.empty():
            return "buffer"
        if self.response:
            return "completed"
        return "pending"

    @property
    def is_pending(self):
        return bool(not self.response and not self.error)


# Temporary cache to allow to query tasks results for a short time after they are completed.
class DataCache:
    def __init__(self):
        self._base = dict()
        self._lock: threading.Lock = threading.Lock()

    def _get_ttl_time(self, ttl: int) -> int:
        return int(time.time()) + ttl

    def _is_expired(self, timestamp: int) -> bool:
        return int(time.time()) >= timestamp

    def clean(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.clean" + ERR_LOCK_FAILED)
        try:
            # Create a list of expired keys to delete
            to_delete = []
            for key in self._base:
                ttl, _ = self._base[key]
                if self._is_expired(ttl):
                    to_delete.append(key)
            # Remove Items
            for key in to_delete:
                (_, val) = self._base[key]
                if isinstance(val, RenderTask):
                    log.debug(f"RenderTask {key} expired. Data removed.")
                elif isinstance(val, SessionState):
                    log.debug(f"Session {key} expired. Data removed.")
                else:
                    log.debug(f"Key {key} expired. Data removed.")
                del self._base[key]
        finally:
            self._lock.release()

    def clear(self) -> None:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.clear" + ERR_LOCK_FAILED)
        try:
            self._base.clear()
        finally:
            self._lock.release()

    def delete(self, key: Hashable) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.delete" + ERR_LOCK_FAILED)
        try:
            if key not in self._base:
                return False
            del self._base[key]
            return True
        finally:
            self._lock.release()

    def keep(self, key: Hashable, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.keep" + ERR_LOCK_FAILED)
        try:
            if key in self._base:
                _, value = self._base.get(key)
                self._base[key] = (self._get_ttl_time(ttl), value)
                return True
            return False
        finally:
            self._lock.release()

    def put(self, key: Hashable, value: Any, ttl: int) -> bool:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.put" + ERR_LOCK_FAILED)
        try:
            self._base[key] = (self._get_ttl_time(ttl), value)
        except Exception as e:
            log.error(traceback.format_exc())
            return False
        else:
            return True
        finally:
            self._lock.release()

    def tryGet(self, key: Hashable) -> Any:
        if not self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
            raise Exception("DataCache.tryGet" + ERR_LOCK_FAILED)
        try:
            ttl, value = self._base.get(key, (None, None))
            if ttl is not None and self._is_expired(ttl):
                log.debug(f"Session {key} expired. Discarding data.")
                del self._base[key]
                return None
            return value
        finally:
            self._lock.release()


manager_lock = threading.RLock()
render_threads = []
current_state = ServerStates.Init
current_state_error: Exception = None
tasks_queue = []
session_cache = DataCache()
task_cache = DataCache()
weak_thread_data = weakref.WeakKeyDictionary()
idle_event: threading.Event = threading.Event()


class SessionState:
    def __init__(self, id: str):
        self._id = id
        self._tasks_ids = []

    @property
    def id(self):
        return self._id

    @property
    def tasks(self):
        tasks = []
        for task_id in self._tasks_ids:
            task = task_cache.tryGet(task_id)
            if task:
                tasks.append(task)
        return tasks

    def put(self, task, ttl=TASK_TTL):
        task_id = id(task)
        self._tasks_ids.append(task_id)
        if not task_cache.put(task_id, task, ttl):
            return False
        while len(self._tasks_ids) > len(render_threads) * 2:
            self._tasks_ids.pop(0)
        return True


def thread_get_next_task():
    from easydiffusion import renderer

    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        log.warn(f"Render thread on device: {renderer.context.device} failed to acquire manager lock.")
        return None
    if len(tasks_queue) <= 0:
        manager_lock.release()
        return None
    task = None
    try:  # Select a render task.
        for queued_task in tasks_queue:
            if queued_task.render_device and renderer.context.device != queued_task.render_device:
                # Is asking for a specific render device.
                if is_alive(queued_task.render_device) > 0:
                    continue  # requested device alive, skip current one.
                else:
                    # Requested device is not active, return error to UI.
                    queued_task.error = Exception(queued_task.render_device + " is not currently active.")
                    task = queued_task
                    break
            if not queued_task.render_device and renderer.context.device == "cpu" and is_alive() > 1:
                # not asking for any specific devices, cpu want to grab task but other render devices are alive.
                continue  # Skip Tasks, don't run on CPU unless there is nothing else or user asked for it.
            task = queued_task
            break
        if task is not None:
            del tasks_queue[tasks_queue.index(task)]
        return task
    finally:
        manager_lock.release()


def thread_render(device):
    global current_state, current_state_error

    from easydiffusion import renderer, model_manager

    try:
        renderer.init(device)

        weak_thread_data[threading.current_thread()] = {
            "device": renderer.context.device,
            "device_name": renderer.context.device_name,
            "alive": True,
        }

        current_state = ServerStates.LoadingModel
        model_manager.load_default_models(renderer.context)

        current_state = ServerStates.Online
    except Exception as e:
        log.error(traceback.format_exc())
        weak_thread_data[threading.current_thread()] = {"error": e, "alive": False}
        return

    while True:
        session_cache.clean()
        task_cache.clean()
        if not weak_thread_data[threading.current_thread()]["alive"]:
            log.info(f"Shutting down thread for device {renderer.context.device}")
            model_manager.unload_all(renderer.context)
            return
        if isinstance(current_state_error, SystemExit):
            current_state = ServerStates.Unavailable
            return
        task = thread_get_next_task()
        if task is None:
            idle_event.clear()
            idle_event.wait(timeout=1)
            continue
        if task.error is not None:
            log.error(task.error)
            task.response = {"status": "failed", "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            continue
        if current_state_error:
            task.error = current_state_error
            task.response = {"status": "failed", "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            continue
        log.info(f"Session {task.task_data.session_id} starting task {id(task)} on {renderer.context.device_name}")
        if not task.lock.acquire(blocking=False):
            raise Exception("Got locked task from queue.")
        try:

            def step_callback():
                global current_state_error

                if (
                    isinstance(current_state_error, SystemExit)
                    or isinstance(current_state_error, StopAsyncIteration)
                    or isinstance(task.error, StopAsyncIteration)
                ):
                    renderer.context.stop_processing = True
                    if isinstance(current_state_error, StopAsyncIteration):
                        task.error = current_state_error
                        current_state_error = None
                        log.info(f"Session {task.task_data.session_id} sent cancel signal for task {id(task)}")

            current_state = ServerStates.LoadingModel
            model_manager.resolve_model_paths(task.task_data)
            model_manager.reload_models_if_necessary(renderer.context, task.task_data)

            current_state = ServerStates.Rendering
            task.response = renderer.make_images(
                task.render_request, task.task_data, task.buffer_queue, task.temp_images, step_callback
            )
            # Before looping back to the generator, mark cache as still alive.
            task_cache.keep(id(task), TASK_TTL)
            session_cache.keep(task.task_data.session_id, TASK_TTL)
        except Exception as e:
            task.error = str(e)
            task.response = {"status": "failed", "detail": str(task.error)}
            task.buffer_queue.put(json.dumps(task.response))
            log.error(traceback.format_exc())
        finally:
            gc(renderer.context)
            task.lock.release()
        task_cache.keep(id(task), TASK_TTL)
        session_cache.keep(task.task_data.session_id, TASK_TTL)
        if isinstance(task.error, StopAsyncIteration):
            log.info(f"Session {task.task_data.session_id} task {id(task)} cancelled!")
        elif task.error is not None:
            log.info(f"Session {task.task_data.session_id} task {id(task)} failed!")
        else:
            log.info(
                f"Session {task.task_data.session_id} task {id(task)} completed by {renderer.context.device_name}."
            )
        current_state = ServerStates.Online


def get_cached_task(task_id: str, update_ttl: bool = False):
    # By calling keep before tryGet, wont discard if was expired.
    if update_ttl and not task_cache.keep(task_id, TASK_TTL):
        # Failed to keep task, already gone.
        return None
    return task_cache.tryGet(task_id)


def get_cached_session(session_id: str, update_ttl: bool = False):
    if update_ttl:
        session_cache.keep(session_id, TASK_TTL)
    session = session_cache.tryGet(session_id)
    if not session:
        session = SessionState(session_id)
        session_cache.put(session_id, session, TASK_TTL)
    return session


def get_devices():
    devices = {
        "all": {},
        "active": {},
    }

    def get_device_info(device):
        if device in ("cpu", "mps"):
            return {"name": device_manager.get_processor_name()}

        mem_free, mem_total = torch.cuda.mem_get_info(device)
        mem_free /= float(10**9)
        mem_total /= float(10**9)

        return {
            "name": torch.cuda.get_device_name(device),
            "mem_free": mem_free,
            "mem_total": mem_total,
            "max_vram_usage_level": device_manager.get_max_vram_usage_level(device),
        }

    # list the compatible devices
    cuda_count = torch.cuda.device_count()
    for device in range(cuda_count):
        device = f"cuda:{device}"
        if not device_manager.is_device_compatible(device):
            continue

        devices["all"].update({device: get_device_info(device)})

    if device_manager.is_mps_available():
        devices["all"].update({"mps": get_device_info("mps")})

    devices["all"].update({"cpu": get_device_info("cpu")})

    # list the activated devices
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        raise Exception("get_devices" + ERR_LOCK_FAILED)
    try:
        for rthread in render_threads:
            if not rthread.is_alive():
                continue
            weak_data = weak_thread_data.get(rthread)
            if not weak_data or not "device" in weak_data or not "device_name" in weak_data:
                continue
            device = weak_data["device"]
            devices["active"].update({device: get_device_info(device)})
    finally:
        manager_lock.release()

    return devices


def is_alive(device=None):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        raise Exception("is_alive" + ERR_LOCK_FAILED)
    nbr_alive = 0
    try:
        for rthread in render_threads:
            if device is not None:
                weak_data = weak_thread_data.get(rthread)
                if weak_data is None or not "device" in weak_data or weak_data["device"] is None:
                    continue
                thread_device = weak_data["device"]
                if thread_device != device:
                    continue
            if rthread.is_alive():
                nbr_alive += 1
        return nbr_alive
    finally:
        manager_lock.release()


def start_render_thread(device):
    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        raise Exception("start_render_thread" + ERR_LOCK_FAILED)
    log.info(f"Start new Rendering Thread on device: {device}")
    try:
        rthread = threading.Thread(target=thread_render, kwargs={"device": device})
        rthread.daemon = True
        rthread.name = THREAD_NAME_PREFIX + device
        rthread.start()
        render_threads.append(rthread)
    finally:
        manager_lock.release()
    timeout = DEVICE_START_TIMEOUT
    while not rthread.is_alive() or not rthread in weak_thread_data or not "device" in weak_thread_data[rthread]:
        if rthread in weak_thread_data and "error" in weak_thread_data[rthread]:
            log.error(f"{rthread}, {device}, error: {weak_thread_data[rthread]['error']}")
            return False
        if timeout <= 0:
            return False
        timeout -= 1
        time.sleep(1)
    return True


def stop_render_thread(device):
    try:
        device_manager.validate_device_id(device, log_prefix="stop_render_thread")
    except:
        log.error(traceback.format_exc())
        return False

    if not manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT):
        raise Exception("stop_render_thread" + ERR_LOCK_FAILED)
    log.info(f"Stopping Rendering Thread on device: {device}")

    try:
        thread_to_remove = None
        for rthread in render_threads:
            weak_data = weak_thread_data.get(rthread)
            if weak_data is None or not "device" in weak_data or weak_data["device"] is None:
                continue
            thread_device = weak_data["device"]
            if thread_device == device:
                weak_data["alive"] = False
                thread_to_remove = rthread
                break
        if thread_to_remove is not None:
            render_threads.remove(rthread)
            return True
    finally:
        manager_lock.release()

    return False


def update_render_threads(render_devices, active_devices):
    devices_to_start, devices_to_stop = device_manager.get_device_delta(render_devices, active_devices)
    log.debug(f"devices_to_start: {devices_to_start}")
    log.debug(f"devices_to_stop: {devices_to_stop}")

    for device in devices_to_stop:
        if is_alive(device) <= 0:
            log.debug(f"{device} is not alive")
            continue
        if not stop_render_thread(device):
            log.warn(f"{device} could not stop render thread")

    for device in devices_to_start:
        if is_alive(device) >= 1:
            log.debug(f"{device} already registered.")
            continue
        if not start_render_thread(device):
            log.warn(f"{device} failed to start.")

    if is_alive() <= 0:  # No running devices, probably invalid user config.
        raise EnvironmentError(
            'ERROR: No active render devices! Please verify the "render_devices" value in config.json'
        )

    log.debug(f"active devices: {get_devices()['active']}")


def shutdown_event():  # Signal render thread to close on shutdown
    global current_state_error
    current_state_error = SystemExit("Application shutting down.")


def render(render_req: GenerateImageRequest, task_data: TaskData):
    current_thread_count = is_alive()
    if current_thread_count <= 0:  # Render thread is dead
        raise ChildProcessError("Rendering thread has died.")

    # Alive, check if task in cache
    session = get_cached_session(task_data.session_id, update_ttl=True)
    pending_tasks = list(filter(lambda t: t.is_pending, session.tasks))
    if current_thread_count < len(pending_tasks):
        raise ConnectionRefusedError(
            f"Session {task_data.session_id} already has {len(pending_tasks)} pending tasks out of {current_thread_count}."
        )

    new_task = RenderTask(render_req, task_data)
    if session.put(new_task, TASK_TTL):
        # Use twice the normal timeout for adding user requests.
        # Tries to force session.put to fail before tasks_queue.put would.
        if manager_lock.acquire(blocking=True, timeout=LOCK_TIMEOUT * 2):
            try:
                tasks_queue.append(new_task)
                idle_event.set()
                return new_task
            finally:
                manager_lock.release()
    raise RuntimeError("Failed to add task to cache.")

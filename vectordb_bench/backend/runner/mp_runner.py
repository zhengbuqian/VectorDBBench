import time
import traceback
import concurrent
import multiprocessing as mp
import logging
from typing import Iterable
import numpy as np
import random
from ..clients import api
from .. import utils
from ... import config
from concurrent.futures import ThreadPoolExecutor


NUM_PER_BATCH = config.NUM_PER_BATCH
log = logging.getLogger(__name__)


class MultiProcessingSearchRunner:
    """ multiprocessing search runner

    Args:
        k(int): search topk, default to 100
        concurrency(Iterable): concurrencies, default [1, 5, 10, 15, 20, 25, 30, 35]
        duration(int): duration for each concurency, default to 30s
    """
    def __init__(
        self,
        db: api.VectorDB,
        test_data: np.ndarray,
        k: int = 100,
        filters: dict | None = None,
        concurrencies: Iterable[int] = [35], #20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100
        duration: int = 3600,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.concurrencies = concurrencies
        self.duration = duration

        self.test_data = utils.SharedNumpyArray(test_data)
        log.info(f"test dataset columns: {len(test_data)}")

    def search1(self, test_np: utils.SharedNumpyArray, q: mp.Queue, cond: mp.Condition, process_id: int = 0) -> tuple[int, float]:
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        with self.db.init():
            test_data = test_np.read().tolist()
            num, idx = len(test_data) / self.nq, 0

            start_time = time.perf_counter()
            count = 0
            while time.perf_counter() < start_time + self.duration:
                try:
                    self.db.search_multiple_embedding(
                        test_data[idx:idx+self.nq],
                        self.k,
                        self.filters,
                    )
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    traceback.print_exc(chain=True)
                    raise e from None

                count += self.nq
                # loop through the test data
                idx = idx + self.nq if idx < num - self.nq else 0

        total_dur = round(time.perf_counter() - start_time, 4)

        return (count, total_dur)

    def search2(self, test_np: utils.SharedNumpyArray, q: mp.Queue, cond: mp.Condition, process_id: int = 0) -> tuple[int, float]:
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        with self.db.init():
            test_data = test_np.read().tolist()
            num, idx = len(test_data) / self.nq, 0

            start_time = time.perf_counter()
            count = 0
            intervals = [0.05, 0.045, 0.04, 0.035] # [, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.03, 0.028, 0.026, 0.024] # [] # ,   0.0275, 0.025, 0.0225, 0.02
            previous_interval = 0
            window = 180
            with ThreadPoolExecutor() as executor:
                # while time.perf_counter() < start_time + self.duration:
                while time.perf_counter() < start_time + window * len(intervals):
                    # s = time.perf_counter()
                    interval = intervals[int(int(time.perf_counter() - start_time) / window)]
                    futures = []
                    if interval != previous_interval:
                        if process_id == 0:
                            log.info(f"Now using interval {interval}, sending ~{int(1 / interval)} requests per second(total is {35 * int(1 / interval)})")
                        previous_interval = interval
                        [fut.result() for fut in futures]
                        futures = []
                        time.sleep(10)
                    try:
                        time.sleep(interval)
                        fut = executor.submit(self.db.search_multiple_embedding,
                            test_data[idx:idx+self.nq],
                            self.k,
                            self.filters,
                        )
                        futures.append(fut)
                    except Exception as e:
                        log.warning(f"VectorDB search_embedding error: {e}")
                        traceback.print_exc(chain=True)
                        raise e from None

                    count += self.nq
                    # loop through the test data
                    idx = idx + self.nq if idx < num - self.nq else 0

        total_dur = round(time.perf_counter() - start_time, 4)

        return (count, total_dur)

    @staticmethod
    def get_mp_context():
        mp_start_method = "spawn"
        log.debug(f"MultiProcessingSearchRunner get multiprocessing start method: {mp_start_method}")
        return mp.get_context(mp_start_method)

    def _run_all_concurrencies_mem_efficient(self) -> float:
        # skip = True
        skip = False
        if skip:
            log.info("Skip concurrent run")
            return
        max_qps = 0
        try:
            for conc in self.concurrencies:
                # self.nq = 32 // conc
                self.nq = 1
                with mp.Manager() as m:
                    q, cond = m.Queue(), m.Condition()
                    with concurrent.futures.ProcessPoolExecutor(mp_context=self.get_mp_context(), max_workers=conc) as executor:
                        log.info(f"Start search {self.duration}s in concurrency {conc}, filters: {self.filters}")
                        future_iter = [executor.submit(self.search1, self.test_data, q, cond, i) for i in range(conc)]
                        # Sync all processes
                        while q.qsize() < conc:
                            sleep_t = conc if conc < 10 else 10
                            time.sleep(sleep_t)

                        with cond:
                            cond.notify_all()
                            log.info(f"Syncing all process and start concurrency search, concurrency={conc}")

                        start = time.perf_counter()
                        all_count = sum([r.result()[0] for r in future_iter])
                        cost = time.perf_counter() - start

                        qps = round(all_count / cost, 4)
                        log.info(f"End search in concurrency {conc}: dur={cost}s, total_count={all_count}, qps={qps}")

                if qps > max_qps:
                    max_qps = qps
                    log.info(f"Update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(f"Fail to search all concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}")
            traceback.print_exc()

            # No results available, raise exception
            if max_qps == 0.0:
                raise e from None

        finally:
            self.stop()

        return max_qps

    def run(self) -> float:
        """
        Returns:
            float: largest qps
        """
        return self._run_all_concurrencies_mem_efficient()

    def stop(self) -> None:
        if self.test_data:
            self.test_data.unlink()
            self.test_data = None

import asyncio
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Awaitable, Callable

logger = logging.getLogger(__name__)

IngestFn = Callable[[str, str, str, str, Optional[str], str], Awaitable[None]]

@dataclass(order=True)
class QueueItem:
    priority: int
    seq: int
    upload_id: str = field(compare=False)
    file_path: str = field(compare=False)
    title: str = field(compare=False)
    doc_type: str = field(compare=False)
    ocr_provider: Optional[str] = field(compare=False)
    file_hash: str = field(compare=False, default="")


class ProcessingQueue:
    def __init__(self, max_concurrent: int = 3):
        self._queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue()
        self._ingest_fn: Optional[IngestFn] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._worker_task: Optional[asyncio.Task] = None
        self._seq = 0
        self._stats = {
            "submitted": 0,
            "processing": 0,
            "done": 0,
            "errors": 0,
        }

    def set_ingest_fn(self, fn: IngestFn):
        self._ingest_fn = fn

    def submit(
        self,
        upload_id: str,
        file_path: str,
        title: str,
        doc_type: str,
        ocr_provider: Optional[str],
        file_size: int = 0,
        file_hash: str = "",
    ) -> str:
        self._seq += 1
        item = QueueItem(
            priority=file_size,
            seq=self._seq,
            upload_id=upload_id,
            file_path=str(file_path),
            title=title,
            doc_type=doc_type,
            ocr_provider=ocr_provider,
            file_hash=file_hash,
        )
        self._queue.put_nowait(item)
        self._stats["submitted"] += 1
        size_kb = file_size / 1024
        logger.info(
            f"Queue: submitted '{title}' ({size_kb:.1f} KB), "
            f"priority={file_size}, position in queue={self._queue.qsize()}"
        )
        return upload_id

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(f"ProcessingQueue worker started (max_concurrent={self._semaphore._value})")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("ProcessingQueue worker stopped")

    async def _worker_loop(self):
        while True:
            item = await self._queue.get()
            asyncio.create_task(self._process_item(item))

    async def _process_item(self, item: QueueItem):
        async with self._semaphore:
            self._stats["processing"] += 1
            qsize = self._queue.qsize()
            logger.info(
                f"Queue: processing '{item.title}' ({item.priority} bytes), "
                f"remaining in queue: {qsize}"
            )
            try:
                if self._ingest_fn:
                    await self._ingest_fn(
                        item.upload_id,
                        item.file_path,
                        item.title,
                        item.doc_type,
                        item.ocr_provider,
                        item.file_hash,
                    )
                self._stats["done"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.exception(f"Queue: error processing '{item.title}': {e}")
            finally:
                self._stats["processing"] -= 1

    def get_status(self) -> dict:
        pending = []
        temp_items = []
        while not self._queue.empty():
            try:
                it = self._queue.get_nowait()
                temp_items.append(it)
                pending.append({
                    "upload_id": it.upload_id,
                    "title": it.title,
                    "size_bytes": it.priority,
                    "position": len(pending) + 1,
                })
            except asyncio.QueueEmpty:
                break
        for it in temp_items:
            self._queue.put_nowait(it)

        return {
            "queue_size": self._queue.qsize(),
            "max_concurrent": self._semaphore._value,
            "stats": dict(self._stats),
            "pending": pending,
        }


_queue: Optional[ProcessingQueue] = None


def get_processing_queue() -> ProcessingQueue:
    global _queue
    if _queue is None:
        _queue = ProcessingQueue(max_concurrent=3)
    return _queue

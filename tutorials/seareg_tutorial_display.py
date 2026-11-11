"""Notebook progress presentation only; no model, data or training operations."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import io
import re
import sys


_PROGRESS = re.compile(r'\s*\d{1,3}%\|[^\r\n]*\|\s*\d+/\d+\s*\[[^\r\n]*\]\s*')
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_MULTISLICE_PROGRESS = re.compile(
    rf'\s*EPOCH (?P<epoch>\d+)/(?P<total>\d+) loss=(?P<loss>{_NUMBER}) '
    rf'seconds=(?P<seconds>{_NUMBER}) optimizer_steps=\d+ propagations=\d+\s*'
)
_ROUTINE_LOG = re.compile(
    r'\s*(?:COMMAND: env .+ -m seareg\.jobs --worker .+'
    r'|SHAPE: \[\d+,\s*\d+\] GPU: (?:None|\d+)'
    r'|CHECKPOINT epoch=\d+ path=.+'
    r'|COMPLETE epochs=\d+ optimizer_steps=\d+ output=.+'
    r'|COMPLETE (?:(?:[A-Za-z]:)?[\\/]|[^\s\\/]+[\\/]).+)\s*'
)
_SINGLE_EPOCH = re.compile(
    rf'\s*EPOCH epoch=\d+ optimizer_steps=\d+ loss={_NUMBER} bpr={_NUMBER} '
    rf'reg={_NUMBER} reg_dna={_NUMBER} seconds={_NUMBER}\s*'
)


def routine_training_line(line):
    """Recognize native management records already retained in worker logs."""
    if _ROUTINE_LOG.fullmatch(line) or _SINGLE_EPOCH.fullmatch(line):
        return True
    prefix = 'TRAINING PROTOCOL: '
    if line.startswith(prefix):
        try:
            config = json.loads(line[len(prefix):])
        except ValueError:
            return False
        return isinstance(config, dict) and {'epochs', 'device', 'seed'} <= config.keys()
    return False


class ProgressStream:
    """Parse fragmented CR/newline updates while passing warnings through."""

    def __init__(self, original, publish, *, pattern=_PROGRESS, publish_on_flush=True,
                 transform=None, suppress=None, before_passthrough=None):
        self.original = original
        self.publish = publish
        self.pattern = pattern
        self.publish_on_flush = publish_on_flush
        self.transform = transform
        self.suppress = suppress
        self.before_passthrough = before_passthrough
        self.pending = ''
        self.last_was_progress = False
        self.unterminated_passthrough = False

    def _emit(self, line, delimiter=''):
        if self.pattern is not None and self.pattern.fullmatch(line):
            text = line
            self.publish(self.transform(text) if self.transform else text)
            self.last_was_progress = True
        elif self.suppress is not None and self.suppress(line):
            self.last_was_progress = True
        elif line or (delimiter == '\n' and not self.last_was_progress):
            if self.before_passthrough is not None:
                self.before_passthrough()
            self.unterminated_passthrough = bool(line) and delimiter != '\n'
            self.original.write(line + delimiter)
            self.last_was_progress = False

    def write(self, text):
        self.pending += text
        while True:
            separator = re.search(r'[\r\n]', self.pending)
            if separator is None:
                break
            line = self.pending[:separator.start()]
            delimiter = self.pending[separator.start()]
            self.pending = self.pending[separator.end():]
            self._emit(line, delimiter)
        return len(text)

    def flush(self):
        if self.publish_on_flush and self.pattern is not None and self.pattern.fullmatch(self.pending):
            line = self.pending
            self.pending = ''
            self._emit(line)
        self.original.flush()

    def finish(self, *, end_line=False):
        self.flush()
        if self.pending:
            line = self.pending
            self.pending = ''
            self._emit(line)
        if end_line and self.unterminated_passthrough:
            self.unterminated_passthrough = False
            self.original.write('\n')
        self.original.flush()


class _NativePublisher:
    def __init__(self, stream):
        self.stream = stream
        self.active = False

    def __call__(self, text):
        self.stream.write('\r' + text)
        self.stream.flush()
        self.active = True

    def newline(self):
        if self.active:
            self.active = False
            self.stream.write('\n')
            self.stream.flush()

    finish = newline


def notebook_publisher(stream=None):
    """Write native carriage-return progress to the notebook stderr stream."""
    return _NativePublisher(sys.stderr if stream is None else stream)


class _MultislicePublisher:
    def __init__(self, stream, *, total=None, publish=None):
        self.stream = stream
        self.publish = publish
        self.target = stream if publish is None else io.StringIO()
        self.bar = None
        self.active = False
        if total is not None:
            self._start(total)

    def _start(self, total):
        from tqdm import tqdm
        self.bar = tqdm(total=total, file=self.target, ascii=False)
        self.active = True

    def __call__(self, line):
        match = _MULTISLICE_PROGRESS.fullmatch(line)
        epoch, total = int(match['epoch']), int(match['total'])
        if total <= 0 or not 0 <= epoch <= total or (self.bar is not None and self.bar.total != total):
            if self.publish is not None:
                self.publish(line)
            else:
                self.newline()
                self.stream.write(line + '\n')
            return
        if self.bar is None:
            self._start(total)
        self.bar.update(epoch - self.bar.n)
        self.active = True
        if self.publish is not None:
            self.publish(str(self.bar))

    def newline(self):
        if self.active and self.publish is None:
            self.active = False
            self.stream.write('\n')
            self.stream.flush()

    def finish(self):
        if self.bar is not None:
            self.bar.close()


@contextmanager
def single_line_training_progress(*, publish=None):
    """Show native tqdm once; keep management records in the worker log files."""
    publish = notebook_publisher() if publish is None else publish
    before = getattr(publish, 'newline', None)
    stream = ProgressStream(sys.stderr, publish, suppress=routine_training_line, before_passthrough=before)
    logs = ProgressStream(sys.stdout, publish, pattern=None, suppress=routine_training_line, before_passthrough=before)
    with redirect_stderr(stream), redirect_stdout(logs):
        try:
            yield
        finally:
            stream.finish()
            logs.finish()
            if hasattr(publish, 'finish'):
                publish.finish()


@contextmanager
def single_line_multislice_training_progress(*, total=None, publish=None):
    """Show native multislice epochs as one bar without changing worker logs."""
    publisher = _MultislicePublisher(sys.stderr, total=total, publish=publish)
    stream = ProgressStream(
        sys.stdout, publisher,
        pattern=_MULTISLICE_PROGRESS,
        publish_on_flush=False,
        suppress=routine_training_line,
        before_passthrough=publisher.newline,
    )
    logs = ProgressStream(sys.stderr, publisher, pattern=None, suppress=routine_training_line,
                          before_passthrough=publisher.newline)
    with redirect_stdout(stream), redirect_stderr(logs):
        try:
            yield
        finally:
            stream.finish(end_line=publish is None)
            logs.finish(end_line=publish is None)
            publisher.finish()

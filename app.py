from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st
from faster_whisper import WhisperModel


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".mp3", ".wav"}


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


class ProcessingCanceled(Exception):
    pass


class ProgressUI:
    def __init__(self, progress_bar, stage_placeholder, log_placeholder, time_placeholder) -> None:
        self.progress_bar = progress_bar
        self.stage_placeholder = stage_placeholder
        self.log_placeholder = log_placeholder
        self.time_placeholder = time_placeholder
        self.started_at = time.monotonic()
        self.logs: list[str] = []
        self.last_percent = 0

    def set_stage(self, stage: str, percent: int, log_message: str | None = None) -> None:
        self.last_percent = max(self.last_percent, min(percent, 100))
        self.stage_placeholder.info(f"현재 단계: {stage}")
        self.progress_bar.progress(self.last_percent, text=stage)
        if log_message:
            self.log(log_message)
        self.update_time(self.last_percent / 100)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        self.log_placeholder.text_area(
            "처리 로그",
            value="\n".join(self.logs[-120:]),
            height=240,
            disabled=True,
        )

    def update_transcription(self, fraction: float | None, segment_count: int) -> None:
        if fraction is None:
            percent = min(90, 55 + segment_count // 3)
        else:
            percent = 55 + int(max(0.0, min(fraction, 1.0)) * 35)

        percent = max(self.last_percent, min(percent, 90))
        self.last_percent = percent
        self.progress_bar.progress(percent, text="자막 생성 중")
        self.update_time(None if fraction is None else percent / 100)

        if segment_count == 1 or segment_count % 10 == 0:
            if fraction is None:
                self.log(f"transcription running - {segment_count} segments generated")
            else:
                self.log(f"transcription {int(fraction * 100)}% complete - {segment_count} segments generated")

    def update_time(self, progress_fraction: float | None = None) -> None:
        elapsed = time.monotonic() - self.started_at
        remaining_text = "계산 중"
        if progress_fraction and progress_fraction > 0.03:
            remaining = max(0.0, elapsed / progress_fraction - elapsed)
            remaining_text = format_duration(remaining)
        self.time_placeholder.caption(
            f"경과 시간: {format_duration(elapsed)} | 예상 남은 시간: {remaining_text}"
        )


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다. Linux에서 `sudo apt install ffmpeg`로 설치해 주세요.")


def save_uploaded_file(uploaded_file, output_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    input_path = output_dir / f"input{suffix}"
    with input_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return input_path


def extract_wav(input_path: Path, output_dir: Path) -> Path:
    wav_path = output_dir / "audio_16k_mono.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return wav_path


def get_audio_duration(path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except Exception:
        return None
    return duration if duration > 0 else None


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, preferred_device: str = "cpu") -> tuple[WhisperModel, str]:
    if preferred_device == "cuda":
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16"), "cuda"
        except Exception:
            pass

    return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


def transcribe_audio(
    wav_path: Path,
    model_size: str,
    preferred_device: str = "cpu",
    progress_ui: ProgressUI | None = None,
) -> tuple[list[SubtitleSegment], str]:
    audio_duration = get_audio_duration(wav_path)
    if progress_ui and audio_duration:
        progress_ui.log(f"audio duration detected: {format_duration(audio_duration)}")

    if progress_ui:
        progress_ui.set_stage("음성 인식 모델 로딩 중", 45, "loading faster-whisper model")
    model, active_device = load_model(model_size, preferred_device)

    try:
        if progress_ui:
            progress_ui.set_stage("자막 생성 중", 55, "transcription started")
        segments = collect_transcription_segments(model, wav_path, audio_duration, progress_ui)
    except Exception:
        if active_device == "cpu":
            raise
        if progress_ui:
            progress_ui.log("GPU transcription failed; retrying with CPU/int8")
            progress_ui.set_stage("음성 인식 모델 로딩 중", 45, "loading CPU/int8 fallback model")
        model, active_device = load_model(model_size, "cpu")
        if progress_ui:
            progress_ui.set_stage("자막 생성 중", 55, "CPU/int8 transcription started")
        segments = collect_transcription_segments(model, wav_path, audio_duration, progress_ui)

    if progress_ui:
        progress_ui.log("subtitle generation finished")
    return segments, active_device


def collect_transcription_segments(
    model: WhisperModel,
    wav_path: Path,
    audio_duration: float | None,
    progress_ui: ProgressUI | None = None,
) -> list[SubtitleSegment]:
    raw_segments, _ = model.transcribe(
        str(wav_path),
        language="ko",
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )
    segments: list[SubtitleSegment] = []
    for raw_segment in raw_segments:
        check_cancel_requested()
        segment = SubtitleSegment(raw_segment.start, raw_segment.end, clean_text(raw_segment.text))
        segments.append(segment)
        if progress_ui:
            fraction = None
            if audio_duration:
                fraction = min(segment.end / audio_duration, 1.0)
            progress_ui.update_transcription(fraction, len(segments))
    return segments


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\.+$", "", text).strip()
    return text


def split_long_segment(segment: SubtitleSegment, max_chars: int = 42) -> list[SubtitleSegment]:
    if len(segment.text) <= max_chars:
        return [segment]

    words = segment.text.split()
    if len(words) <= 1:
        return [segment]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))

    duration = max(segment.end - segment.start, 0.1)
    chunk_duration = duration / len(chunks)
    return [
        SubtitleSegment(
            start=segment.start + index * chunk_duration,
            end=segment.start + (index + 1) * chunk_duration,
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_display_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes = seconds // 60
    secs = seconds % 60
    if minutes >= 60:
        hours = minutes // 60
        minutes %= 60
        return f"{hours}시간 {minutes}분 {secs}초"
    return f"{minutes}분 {secs}초"


def make_srt(segments: Iterable[SubtitleSegment]) -> str:
    subtitle_blocks = []
    subtitle_index = 1
    for segment in segments:
        for part in split_long_segment(segment):
            if not part.text:
                continue
            subtitle_blocks.append(
                f"{subtitle_index}\n"
                f"{format_srt_time(part.start)} --> {format_srt_time(part.end)}\n"
                f"{part.text}\n"
            )
            subtitle_index += 1
    return "\n".join(subtitle_blocks)


def make_srt_filename(original_name: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", Path(original_name).stem).strip("_") or "subtitle"
    return f"{safe_stem}.srt"


def render_text_download(text: str, file_name: str, label: str, mime: str) -> None:
    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=file_name,
        mime=mime,
        use_container_width=True,
    )


def render_bytes_download(data: bytes, file_name: str, label: str, mime: str) -> None:
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        use_container_width=True,
    )


def append_download_log(logs: list[str], placeholder, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    logs.append(f"[{timestamp}] {message}")
    placeholder.text_area(
        "yt-dlp 로그",
        value="\n".join(logs[-160:]),
        height=280,
        disabled=True,
    )


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"

    size = float(size_bytes)
    for unit in ["KB", "MB", "GB", "TB"]:
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PB"


def find_latest_download(download_dir: Path, started_at: float) -> Path | None:
    candidates = [
        path
        for path in download_dir.glob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= started_at - 2
    ]
    if not candidates:
        candidates = [path for path in download_dir.glob("*.mp4") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def download_video_to_temp(url: str, download_dir: Path, progress_bar, log_placeholder) -> tuple[bytes, str, int]:
    logs: list[str] = []
    started_at = time.time()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bv*[vcodec*=avc1]+ba[ext=m4a]",
        "--merge-output-format",
        "mp4",
        "--newline",
        "--paths",
        str(download_dir),
        "-o",
        "%(title).200B.%(ext)s",
        url,
    ]

    progress_bar.progress(0, text="다운로드 준비 중")
    append_download_log(logs, log_placeholder, 'yt-dlp download started')
    append_download_log(logs, log_placeholder, 'format: bv*[vcodec*=avc1]+ba[ext=m4a], merge: mp4')

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    technical_lines: list[str] = []
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        technical_lines.append(line)
        append_download_log(logs, log_placeholder, line)

        percent_match = re.search(r"(\d+(?:\.\d+)?)%", line)
        if percent_match:
            percent = min(99, max(1, int(float(percent_match.group(1)))))
            progress_bar.progress(percent, text=f"다운로드 중 {percent}%")

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("\n".join(technical_lines[-80:]) or f"yt-dlp exited with code {return_code}")

    downloaded_path = find_latest_download(download_dir, started_at)
    if downloaded_path is None:
        raise RuntimeError("다운로드는 완료되었지만 저장된 mp4 파일을 찾지 못했습니다.")

    video_bytes = downloaded_path.read_bytes()
    file_name = downloaded_path.name
    file_size = len(video_bytes)
    progress_bar.progress(100, text="다운로드 완료")
    append_download_log(logs, log_placeholder, "download completed")
    return video_bytes, file_name, file_size


def request_cancel() -> None:
    st.session_state.cancel_requested = True


def ensure_processing_state() -> None:
    st.session_state.setdefault("cancel_requested", False)


def check_cancel_requested() -> None:
    if st.session_state.get("cancel_requested"):
        raise ProcessingCanceled("사용자가 작업을 취소했습니다.")


def render_subtitle_tab() -> None:
    st.subheader("자막 생성")
    st.caption("업로드한 파일에서 wav 추출, 음성 인식, SRT 생성을 순서대로 실행합니다.")

    with st.expander("자막 설정", expanded=True):
        model_size = st.selectbox(
            "Whisper 모델",
            ["small", "medium", "large-v3"],
            index=1,
            help="Korean accuracy is usually better with medium or large-v3. Use small for low-spec machines.",
        )
        preferred_device = "cuda" if st.checkbox(
            "GPU 사용 시도",
            value=False,
            help="기본값은 CPU/int8입니다. CUDA가 없거나 오류가 나면 자동으로 CPU/int8로 전환합니다.",
        ) else "cpu"
        st.caption("기본 실행 모드: CPU / int8")

    uploaded_file = st.file_uploader(
        "파일 업로드",
        type=["mp4", "mkv", "mov", "mp3", "wav"],
        accept_multiple_files=False,
    )

    if not uploaded_file:
        st.info("mp4, mkv, mov, mp3, wav 파일을 업로드해 주세요.")
        return

    if st.button("자막 생성 시작", type="primary", use_container_width=True):
        st.session_state.cancel_requested = False

        progress = st.progress(0, text="준비 중...")
        stage_placeholder = st.empty()
        time_placeholder = st.empty()
        st.button("처리 취소", on_click=request_cancel, use_container_width=True)
        log_placeholder = st.empty()
        progress_ui = ProgressUI(progress, stage_placeholder, log_placeholder, time_placeholder)

        try:
            progress_ui.set_stage("영상 불러오는 중", 5, "processing started")
            check_ffmpeg()
            progress_ui.log("ffmpeg check completed")
            check_cancel_requested()

            with tempfile.TemporaryDirectory() as temp_dir_name:
                temp_dir = Path(temp_dir_name)

                progress_ui.set_stage("영상 불러오는 중", 10, "loading uploaded media")
                input_path = save_uploaded_file(uploaded_file, temp_dir)
                progress_ui.set_stage("영상 불러오는 중", 15, "media file loaded")
                check_cancel_requested()

                progress_ui.set_stage("wav 추출 중", 20, "ffmpeg wav extraction started")
                wav_path = extract_wav(input_path, temp_dir)
                progress_ui.set_stage("wav 추출 중", 35, "wav extraction completed")
                check_cancel_requested()

                segments, active_device = transcribe_audio(wav_path, model_size, preferred_device, progress_ui)
                if preferred_device == "cuda" and active_device == "cpu":
                    st.warning("CUDA를 사용할 수 없어 CPU/int8 모드로 자동 전환했습니다.")
                else:
                    st.info(f"사용 중인 음성 인식 모드: {active_device}/{'int8' if active_device == 'cpu' else 'float16'}")
                progress_ui.set_stage("자막 생성 중", 90, "transcription completed")
                check_cancel_requested()

                progress_ui.set_stage("SRT 저장 중", 92, "building SRT text")
                srt_text = make_srt(segments)
                check_cancel_requested()

                progress_ui.set_stage("완료", 100, "subtitle generation finished")
                stage_placeholder.success("완료: 아래 다운로드 버튼으로 SRT 파일을 받을 수 있습니다.")

                st.subheader("생성된 자막 미리보기")
                st.text_area("SRT", srt_text[:6000], height=260)
                render_text_download(
                    srt_text,
                    make_srt_filename(uploaded_file.name),
                    "SRT 다운로드",
                    "text/plain",
                )

        except ProcessingCanceled:
            progress_ui.log("processing canceled by user")
            stage_placeholder.warning("작업이 취소되었습니다.")
        except subprocess.CalledProcessError as error:
            progress_ui.log("ffmpeg error occurred")
            st.error("오디오 추출 중 문제가 발생했습니다. 파일 형식이나 ffmpeg 설치 상태를 확인해 주세요.")
            with st.expander("기술 오류 자세히 보기"):
                st.code(error.stderr or str(error))
        except Exception as error:
            progress_ui.log("unexpected error occurred")
            st.error("처리 중 문제가 발생했습니다. 다시 시도하거나 더 작은 파일로 테스트해 주세요.")
            with st.expander("기술 오류 자세히 보기"):
                st.code("".join(traceback.format_exception(error)))


def render_video_download_tab() -> None:
    st.subheader("영상 다운로드")
    st.caption("URL에서 영상을 다운로드합니다. 자막 생성 도구와는 연결되지 않는 독립 기능입니다.")

    url = st.text_input("URL 입력", placeholder="https://...")
    if st.button("다운로드 시작", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("다운로드할 URL을 입력해 주세요.")
            return

        progress = st.progress(0, text="대기 중")
        log_placeholder = st.empty()

        try:
            with tempfile.TemporaryDirectory() as temp_dir_name:
                video_bytes, file_name, file_size = download_video_to_temp(
                    url.strip(),
                    Path(temp_dir_name),
                    progress,
                    log_placeholder,
                )
            st.success("다운로드가 준비되었습니다. 아래 버튼을 눌러 파일을 저장하세요.")
            st.write(f"파일명: `{file_name}`")
            st.write(f"파일 크기: `{format_file_size(file_size)}`")
            render_bytes_download(video_bytes, file_name, "영상 다운로드", "video/mp4")

        except FileNotFoundError:
            st.error("yt-dlp를 실행할 수 없습니다. `pip install -r requirements.txt`를 먼저 실행해 주세요.")
        except Exception as error:
            st.error("영상 다운로드 중 문제가 발생했습니다. URL이 올바른지, 해당 사이트에서 다운로드가 가능한지 확인해 주세요.")
            with st.expander("기술 오류 자세히 보기"):
                st.code("".join(traceback.format_exception(error)))


def main() -> None:
    st.set_page_config(page_title="로컬 영상 도구", page_icon="🎧", layout="centered")
    ensure_processing_state()
    st.title("로컬 영상 도구")
    st.caption("자막 생성과 영상 다운로드를 한 앱에서 따로 실행합니다.")

    subtitle_tab, download_tab = st.tabs(["자막 생성", "영상 다운로드"])
    with subtitle_tab:
        render_subtitle_tab()
    with download_tab:
        render_video_download_tab()


if __name__ == "__main__":
    main()

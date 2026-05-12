# 로컬 영상 도구

Streamlit에서 두 가지 로컬 도구를 독립적으로 실행합니다.

- `자막 생성`: 업로드 파일에서 WAV 추출, 한국어 음성 인식, SRT 생성
- `영상 다운로드`: URL로 영상을 다운로드

두 기능은 서로 연결되지 않습니다. 다운로드한 영상을 자동으로 자막 생성에 보내지 않습니다.

## 지원 파일

자막 생성 탭에서 지원하는 업로드 형식입니다.

- mp4
- mkv
- mov
- mp3
- wav

## 설치

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
streamlit run app.py
```

## 자막 생성

`자막 생성` 탭에서 파일을 업로드하고 `자막 생성 시작`을 누릅니다.

처리 단계:

- wav 추출
- 음성 인식
- srt 생성

생성이 끝나면 화면의 `SRT 다운로드` 버튼으로 파일을 받을 수 있습니다. SRT 파일은 로컬 `outputs/` 폴더에 자동 저장되지 않습니다.

기본 실행은 CUDA가 필요 없는 `CPU / int8` 모드입니다. NVIDIA GPU 사용을 시도할 수 있지만, CUDA 라이브러리가 없거나 GPU 실행에 실패하면 앱이 자동으로 `CPU / int8`로 전환합니다.

## 영상 다운로드

`영상 다운로드` 탭에서 URL을 입력하고 `다운로드 시작`을 누릅니다.

앱은 다음 방식으로 `yt-dlp`를 실행합니다.

```bash
yt-dlp -f "bv*[vcodec*=avc1]+ba[ext=m4a]" --merge-output-format mp4
```

다운로드가 끝나면 화면의 `영상 다운로드` 버튼으로 파일을 받을 수 있습니다. 영상 파일은 로컬 `downloads/` 폴더에 자동 저장되지 않습니다.

## 모델 선택 팁

- `medium`: 한국어 정확도와 속도의 균형이 좋습니다.
- `large-v3`: 정확도 우선. 첫 실행 다운로드와 처리 시간이 더 깁니다.
- `small`: 저사양 CPU에서 빠르게 테스트할 때 적합합니다.

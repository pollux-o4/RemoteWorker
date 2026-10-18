# 로컬 기본값 설정

| 설정 원천 | 해석 |
| --- | --- |
| `REMOTEWORKER_CONFIG=/path/to/defaults.json` | 명시한 파일; 누락·잘못된 스키마 → 초기화 오류 |
| `REMOTEWORKER_CONFIG` 미설정 | `$XDG_CONFIG_HOME/remoteworker/defaults.json`, 또는 `~/.config/remoteworker/defaults.json` |
| `REMOTEWORKER_CONFIG=""` | 공개 기본값만 적용 |
| 저장한 워크플로 | 저장된 위젯 입력; 자동 변경 없음 |

## 필드

| 키 | 공개 기본값 |
| --- | --- |
| `server_url` | `http://127.0.0.1:8188` |
| `conditioning_clip`, `qwen_clip`, `qwen_vae` | 미선택 |
| `conditioning_clip_type` | `lumina2` |
| `minimax_clip`, `minimax_video_vae`, `minimax_audio_vae` | 미선택 |
| `prompt_profile` | 일반 예제 프로필 |

```json
{
  "server_url": "http://127.0.0.1:8188",
  "qwen_clip": "text-encoder.safetensors",
  "qwen_vae": "vae.safetensors"
}
```

`prompt_profile`: 전체 프로필 필드 — `id`, `description`, `clip_name`, `clip_type`, `device`, `max_length`, `system_prompt`, `accepted_media`.

적용 시점: Python 모듈 import. 재적용 경계: ComfyUI 프로세스 재시작. 브라우저 초기 조회 주소: 백엔드 노드 스키마. peer 관리 프로필: 기존 ComfyUI 사용자 디렉터리 레지스트리; 기본값 설정과 별도.

# Rust native 실행 계약

| 항목 | 정본 |
| --- | --- |
| 구성 | 단일 Rust crate·단일 `_native` 확장 · 프로세스 내부 모듈 경계 |
| Python | CPython 3.12+ · PyO3 abi3 |
| 공유 파일 플랫폼 | Linux · flock·no-follow dirfd·`/proc/<pid>/fd` |
| 도구체인 | `rust-toolchain.toml` · `Cargo.lock` |
| 개발 빌드 | `python tools/build_native.py --debug` |
| 배포 빌드 | 해당 런타임의 `python install.py` |
| 통합 검사 | `python tools/verify_native.py` |
| wheel | `maturin build --release --locked` · `pyproject.toml` |
| native 누락 | 로컬 빌드·설치 wheel 탐색 후 import 실패 · Python 로직 폴백 없음 |

## 설치·수명주기

- `install.py`: ComfyUI Manager·GenRel 의존성 설치 훅
- `build_native.py`: 빌드 잠금·Cargo locked 빌드·동일 디렉터리 임시 바이너리·fsync·원자 교체·새 Python 프로세스 import 검사
- 바이너리 교체 이후 ComfyUI 재시작 · 기존 프로세스의 기존 바이너리 매핑 유지
- 소스 checkout의 바이너리 우선 · 부재 시 설치된 `remoteworker-native` wheel 사용
- Python adapter: Comfy 등록·thread/context·HTTP·PyTorch/native GPU API · Rust: 입력 계약·핵심 규칙·실행 순서·상태·정리 판단

## 검증 경계

- Rust 단위·컴파일 실패 검사: 잘못된 ID 구성·정책 분기·상태 전이
- Python/JS 계약: 기존 노드 ID·입출력·저장 워크플로·취소·소켓·결과·정리
- 실제 ComfyUI: 격리 포트·독립 base/input/output/temp/user · 운영 큐와 분리
- GPU 실행 무결성: 모델별 native 결과 비교·오디오·영상·타일·원격 파이프라인

- 공유 파일 writer: 검증된 디렉터리·임시 파일 FD 고정 · pathname 변경과 분리 · subprocess 출력 형식 명시

- 공유 디스크 권한: 호출·peer 동일 Unix UID · 요청 디렉터리 0700·파일 0600 · 다른 UID 간 전송은 HTTP tensor 모드

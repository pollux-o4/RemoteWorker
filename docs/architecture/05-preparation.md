# 자동 준비 · CLIP 우선·2 step VAE 전환

## 실행 정책

- [AMD 파이프라인](08-pipeline.md) 활성 시: 다음 CLIP lookahead 유지·실제 VAE 디코딩으로 dummy 웜업 대체·원격 대기 중 다음 작업 실행
- 아래 2 step 웜업 정책: 파이프라인 비활성 host의 실행 경로


| 조건 | 처리 |
| --- | --- |
| 샘플링 3 step 이상 남음 | 다음 conditioning 처리 · VAE 웜업 보류 |
| 샘플링 2 step 이하 남음 | VAE 웜업 준비 |
| 다음 선두 작업의 conditioning 미준비 | 남은 step·VAE 준비 여부와 무관하게 CLIP 먼저 완료 |
| CLIP 중 샘플링 종료 | CLIP 유지 · 현재 VAE 디코딩의 gate 대기 |
| 다음 작업 삭제·선두 변경·입력 변경 | 해당 선계산 취소 · 새 선두 확인 |
| VAE dummy 웜업 중 새 CLIP 요청 | 해당 dummy 웜업 협력 취소 · CLIP 이후 VAE 재준비 |
| 이미 실행 중인 실제 VAE decode | 현재 GPU 연산 유지 · 즉시 선점 제외 |

- 권위 큐: ComfyUI 원본 큐 · 다음 선두 1개만 lookahead
- 판단 근거: native structured progress의 `total - step` · `Admission.vae_steps` 임계값 정본·상태 API/UI 공유
- 최근 CLIP·VAE 처리 시간: 관측용 · ETA 추정·시간 예산 admission 제거
- VAE 이미 준비된 경우: 새 CLIP 실행 시 준비 예약 무효화 · CLIP 후 재준비 또는 실제 decode로 로딩
- 디코딩 진입 장벽: 같은 peer의 미준비 다음 CLIP 우선 처리 · 전경 decode gate 입장 이전
- 삭제된 다음 작업: 현재 영상 실패로 전파 금지 · 취소·재정렬 뒤 선두 재확인
- `GenRelInput`: GenRel 공통 입력 · 문자열/숫자·CPU 이미지/마스크·영상/오디오 로더 · `Resolver.SAFE_TYPES` 허용 · GPU 모델 로딩 제외
- `RequestConditionInput`: GenRel Draft의 CPU 문자열 입력 · `Resolver.SAFE_TYPES` 허용 · name/text → STRING 계약
- 지원 외 입력·다음 encoder 실패·handoff 용량 초과: 상태 표시·일반 전경 재계산 경로 유지
- 수동 warmup 연결: 자동 웜업 선택보다 우선 · 실제 decode의 다음 CLIP 우선 장벽 유지

## 원격 노드의 준비 계약

- 정본: 등록 노드 클래스의 `REMOTE_PREPARATION: Preparation` · 중앙 노드명 허용 목록 없음
- `capabilities.discover`: 노드 등록 매핑 → 역할·실행 대상·입력 adapter 발견
- encoder: 다음 선두의 실제 입력 선계산 · decoder: 현재 작업 첫 원격 VAE의 고정 dummy 준비
- encoder 단독·decoder 단독·혼합 그래프의 동일 정책 경로 · 로컬 `VAEDecode` 제외
- `Preparation.executor`: 네이티브 노드와 분리된 실행 서비스 지정 · 미지정 시 등록 클래스
- `Preparation.inputs`: 저장 그래프 입력 → 실행 서비스 인자 · Qwen `images.image_N` → `image_N`
- 전경·사전 계산: 동일 실행 서비스의 `run` 경유 · 기본값 정규화·peer gate·일회 handoff 공유
- 작업 내 노드 식별: legacy `UNIQUE_ID`·V3 `Hidden.unique_id` → 실행 서비스 `unique_id`
- 선언 없는 노드·미감사 GPU/LLM 조상: 자동 실행 제외 · 일반 전경 처리 유지

## C3 · 실행 협력

```mermaid
flowchart TB
    native["Comfy 큐·실행·progress 이벤트"]
    bridge["EventBridge<br/>클라이언트 전달 보존"]
    coordinator["Coordinator<br/>현재 작업·다음 선두 관측"]
    planner["plan<br/>선두·도달 그래프·미완료 encoder"]
    resolver["Resolver<br/>허용 CPU 조상만 해석"]
    policy["Admission<br/>미준비 CLIP 우선·2 step 경계"]
    clip["다음 실제 conditioning"]
    handoff["Handoffs<br/>작업별 일회 전달"]
    vae["현재 첫 VAE dummy 준비"]
    barrier["실제 decode 진입 장벽<br/>미준비 다음 CLIP 확인"]
    decode["실제 VAE decode"]
    native --> bridge --> coordinator --> planner --> resolver --> policy
    policy -->|미준비 CLIP| clip --> handoff
    policy -->|2 step 이하| vae
    handoff --> barrier
    vae --> barrier --> decode
```

- 구동: 이벤트 wake·이벤트 합치기 · 5초 안전 재확인
- 취소 신호: 백그라운드 준비의 전경 interrupt 소비 금지 · gate 입장 대기 중 전경 취소 확인
- 별도 주기: 원격 요청 history polling · 상태 창 표시 중 2초 조회
- CPU 조상: 이미지 로딩·크기·스케일·해상도 선택·primitive·수식 허용 목록 · 모델/LLM 임의 실행 제외
- PeerGate: peer별 실행 직렬화 · 입장한 CLIP 완료 후 decode · 동일 요청 중복 선계산 방지
- 범위: 이 호출 서버의 다음 선두·동일 peer · 외부 NVIDIA/LLM 작업의 전역 스케줄링 제외

## 콜드 스타트·웜업·실제 처리

| 구분 | 입력·수행 | 범위 |
| --- | --- | --- |
| 콜드 스타트 | 모델 파일 읽기·CPU staging·GPU 배치 | 최초 CLIP/웜업/실제 decode에 포함 |
| VAE 웜업 | 실제 선택 모델 + 고정 zeros latent decode | 결과 폐기 · 실제 출력 대체 없음 |
| 다음 CLIP 준비 | 다음 작업의 실제 prompt·참조 이미지·해상도·길이 | dummy prompt 미사용 |
| 실제 VAE | 현재 sampling latent decode | 영상 우선 · 오디오는 실제 decode 시 로딩 가능 |

- 고정 dummy: H3 영상 `(1,C,2,2,2)` / H3 오디오 `(1,32,2,8)` / 일반 이미지 `(1,C,8,8)`
- I2V 참조 이미지의 VAE 인코딩: CLIP 준비 단계의 필수 연산 · dummy 웜업 시점과 별도
- 8GB 범위: 영상·오디오 VAE 동시 상주 보장 없음
- Handoffs: prompt ID·node ID·resolved signature 일치 시 1회 소비 · `always_refresh` 작업 간 재사용 금지 유지
- RAM 상한: Handoffs 직렬화 256MiB · 초과 예상 텐서 직렬화 생략 · 전체 프로세스 RAM 상한 아님
- RAM 압력: MemAvailable 2GiB 미만·30초 간격 · 폐기 가능 재사용 cache·GC 정리
- 보존: 활성 텐서·대기 Handoffs · 취소/소비/큐 이탈 결과 해제
- 제외: OS page cache drop·swapoff·활성 모델 강제 unload
- 지연 tradeoff: 늦은 다음 CLIP의 완료 대기로 현재 영상 출력 지연 가능 · CLIP→VAE 전환 순서 우선

## C4 · 코드 책임

| 소스 | 소유 책임 |
| --- | --- |
| `preparation/capabilities.py` | `Preparation`·등록 노드 역할 발견·실행 대상·입력 adapter |
| `preparation/integration.py` | Comfy 등록·wrapper·decode 진입 장벽·handoff 소비 |
| `preparation/events.py` | native 이벤트 어댑터·wake·progress 연결 |
| `preparation/coordinator.py` | 준비 순서·decode 장벽·취소 후 재확인 |
| `preparation/plan.py` | 선두 선택·출력 도달 그래프·미완료 encoder 판정 정본 |
| `preparation/invocation.py` | 호출 기본값 정규화·중첩 speculative 컨텍스트·예외 후 복구 |
| `preparation/policy.py` | 부작용 없는 CLIP 우선·2 step 결정 |
| `preparation/progress.py` | native sampler step·total 보관 · 시간 추정 제외 |
| `preparation/scheduler.py` | 준비 예약·step 결정 상태·실행 시간 관측 |
| `preparation/actions.py` | CLIP 선계산·VAE 예약 무효화·dummy 웜업·요청별 취소 이벤트 |
| `preparation/inputs.py` | CPU 조상 허용 목록·순환/깊이 차단 |
| `preparation/handoff.py` | 직렬화 예산·입력 서명·일회 전달 |
| `preparation/gates.py` | peer별 입장 직렬화·전경 입장 우선 |
| `preparation/memory.py` | 가용 RAM·폐기 가능 cache 정리 |
| `core/cancellation.py`·`core/transport.py` | 요청별 협력 취소·전송·history 확인 |
| `vae/warmup.py`·`vae/worker.py` | ready receipt·고정 dummy decode |
| `web/preparation.js`·`scheduler_view.js` | 상태 창·step 구간·다음 CLIP 우선 설명 |

## C4 · 호출 의존성

```mermaid
flowchart TB
    host["integration · ComfyUI Adapter"]
    orchestration["Coordinator<br/>prepare_next() · next_clip_missing()"]
    queue["plan.pending_encoders()<br/>head() · job_graph() · fingerprint()"]
    execution["actions.prefetch()<br/>prepare_vae()"]
    invocation["invocation<br/>call_values() · speculative_call()"]
    host --> orchestration
    orchestration --> queue
    orchestration --> execution
    host --> invocation
    execution --> invocation
```

- 등록 어댑터 역참조 제거 · 실행 계층의 ComfyUI 등록 의존성 제거
- 실행 컨텍스트: ContextVar 토큰 복원 · 중첩 호출·예외 경계 보존
- 선두 계획: 백그라운드 준비·decode 장벽의 동일 후보 판정

## 검증 경계

- 회귀: `tests/test_preparation_contracts.py`·`test_scheduler_policy.py`·`test_scheduler_ui.py`·`test_preparation_inputs.py`·`test_preparation_capabilities.py`
- 실제 GPU: `tools/e2e_scheduler.py` · 2 step 경계·3초 후 추가·2 step에서 추가·취소·선두 변경
- 출력: MiniMax H3 연결 I2V · 96×96·3 frames·MP4 0.100초 ± 0.002초
- 전달: 다음 실행 전 준비 완료·1회 소비 · 종료 Handoffs 0
- 범위 제외: 긴 영상·고해상도 성능/품질 비교·실제 decode GPU 커널의 즉시 선점

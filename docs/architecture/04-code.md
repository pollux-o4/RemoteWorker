# C4 · RemoteWorker 코드 라우팅

## 호출·전송

| 진입점 | 소스 |
| --- | --- |
| 요청 수명주기 | [execution.py](../../remoteworker/core/execution.py) |
| HTTP 전송 | [transport.py](../../remoteworker/core/transport.py) |
| 프로토콜 | [protocol.py](../../remoteworker/core/protocol.py) |
| 결과 캐시 | [result_cache.py](../../remoteworker/core/result_cache.py) |
| Qwen adapter | [service.py](../../remoteworker/qwen/service.py) |

## 준비·파이프라인

| 진입점 | 소스 |
| --- | --- |
| 역할 발견 | [capabilities.py](../../remoteworker/preparation/capabilities.py) |
| 준비 조정 | [coordinator.py](../../remoteworker/preparation/coordinator.py) |
| native worker 연결 | [host.py](../../remoteworker/pipeline/host.py) |
| 실행 슬롯 | [slots.py](../../remoteworker/pipeline/slots.py) |
| MODEL/LoRA 재사용 | [reuse.py](../../remoteworker/pipeline/reuse.py) |
| CPU 미리보기 | [previews.py](../../remoteworker/pipeline/previews.py) |

## VAE

| 진입점 | 소스 |
| --- | --- |
| 호출 서비스 | [service.py](../../remoteworker/vae/service.py) |
| worker | [worker.py](../../remoteworker/vae/worker.py) |
| 직접·공유 선택 | [delivery.py](../../remoteworker/vae/delivery.py) |
| 텐서 codec | [codec.py](../../remoteworker/vae/codec.py) |
| 공유 텐서 | [shared_tensor.py](../../remoteworker/vae/shared_tensor.py) |
| 공유 파일 | [shared_store.py](../../remoteworker/vae/shared_store.py) |
| 미디어 결합 | [shared_video.py](../../remoteworker/vae/shared_video.py) |
| 노드 schema | [schema.py](../../remoteworker/vae/schema.py) |
| 웜업 | [warmup.py](../../remoteworker/vae/warmup.py) |
| 결과 routes | [routes.py](../../remoteworker/vae/routes.py) |

## 프런트엔드

| 진입점 | 소스 |
| --- | --- |
| 선택 목록 | [remote_choices.js](../../web/remote_choices.js) |
| Qwen 선택 | [remote_qwen_image21.js](../../web/remote_qwen_image21.js) |
| VAE 조작 | [remote_vae.js](../../web/remote_vae.js) |
| VAE 전달 | [vae_delivery.js](../../web/vae_delivery.js) |
| 스케줄 상태 | [scheduler_view.js](../../web/scheduler_view.js) |

## 다른 저장소

- [GenRel C4](../../../../docs/아키텍처/04-code.md)

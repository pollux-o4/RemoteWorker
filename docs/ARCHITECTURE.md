# RemoteWorker 아키텍처

| 단계 | 범위 | 문서 |
| --- | --- | --- |
| C1 · Context | 사용자·ComfyUI·peer·저장소 | [시스템 맥락](architecture/01-context.md) |
| C2 · Container | 브라우저·호출 측·peer 프로세스 | [실행 구조](architecture/02-containers.md) |
| C3 · Component | Rust 모듈러 모놀리스·Python adapter·브라우저 | [컴포넌트](architecture/03-components.md) |
| C4 · Code | 컴포넌트별 코드 진입점 | [소스 라우팅](architecture/04-code.md) |

- [native 빌드·배포](contracts/native-runtime.md)
- [준비·메모리](contracts/preparation.md)
- [실행 파이프라인](contracts/pipeline.md)
- [공유 전송](contracts/shared-video.md)
- [파일 결합·최종화](contracts/file-finalization.md)

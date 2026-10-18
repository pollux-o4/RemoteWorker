# C3 · 컴포넌트

```mermaid
flowchart TB
    subgraph Browser[브라우저 확장]
        direction TB
        Entry[확장 lifecycle]
        Choices[모델 조회·선택]
        Profiles[프로필 상태·편집기]
        Sockets[동적 소켓·워크플로 마이그레이션]
        Entry --> Choices
        Entry --> Profiles
        Entry --> Sockets
        Profiles --> Sockets
    end
    subgraph Caller[호출 측 Python 프로세스]
        direction TB
        Adapter[모델 어댑터·workflow builder]
        Execution[요청 수명주기]
        Transport[Comfy HTTP 어댑터]
        Protocol[검증·artifact 계약]
        Cache[용량 제한 byte 결과 캐시]
        Defaults[외부 기본값 로더]
        Adapter --> Execution --> Transport
        Adapter --> Protocol
        Adapter --> Cache
        Adapter --> Defaults
        Execution --> Protocol
        Discovery[동일 origin 조회·프로필 프록시] --> Transport
    end
    subgraph Peer[peer Python 프로세스]
        direction TB
        Host[Comfy 그래프 실행기]
        Workers[인코딩·생성 worker]
        IO[conditioning·latent·텍스트 저장기]
        Registry[프로필 저장소·CRUD 라우트]
        Cleanup[요청 입력 정리 라우트]
        Host --> Workers --> IO
        PeerDefaults[로컬 기본값 로더]
        Registry --> PeerDefaults
    end
    Choices --> Discovery
    Profiles --> Discovery
    Transport -->|Comfy API| Host
    Transport --> Registry
    Transport --> Cleanup
```

| 컴포넌트 | 소유 코드 |
| --- | --- |
| 확장 lifecycle | `web/remote_conditioning.js`, `web/remote_qwen_image21.js` |
| 조회·모델 선택 | `web/remote_choices.js` |
| 프로필 상태·편집기 | `web/remote_profiles.js`, `web/remote_profile_dialog.js` |
| 그래프 소켓·마이그레이션 | `web/remote_sockets.js` |
| MiniMax 미디어·workflow | `remoteworker/minimax/media.py`, `remoteworker/minimax/reference_workflow.py` |
| 노드·라우트 조립 | `remoteworker/registry.py` |
| 요청 수명주기·HTTP | `remoteworker/core/execution.py`, `remoteworker/core/transport.py` |
| 공유 검증·미디어·artifact | `remoteworker/core/protocol.py`, `remoteworker/core/media.py`, `remoteworker/core/artifacts.py` |
| 캐시 | `remoteworker/core/result_cache.py` |
| 조회·정리 라우트 | `remoteworker/integration/discovery_routes.py` |
| 프로필 저장·프록시 | `remoteworker/prompt/profiles.py`, `remoteworker/prompt/transport.py`, `remoteworker/integration/profile_routes.py` |
| artifact IO | `remoteworker/integration/artifact_io.py` |

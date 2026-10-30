"""Profile endpoints layered on the shared ComfyUI HTTP transport."""

from typing import Any

from ..core.transport import HttpComfyApi
from .profiles import PromptEnhancerProfile, validate_profile_payload


class PromptComfyApi(HttpComfyApi):
    def list_prompt_enhancer_profiles(self) -> list[dict[str, Any]]:
        response = self._json("/remote-prompt-enhancer/profiles")
        from .._native import profile_response_native

        return profile_response_native(response, "list")

    def get_prompt_enhancer_profile(self, profile_id: str) -> PromptEnhancerProfile:
        from .._native import profile_select_native

        profiles = [
            validate_profile_payload(payload)
            for payload in self.list_prompt_enhancer_profiles()
        ]
        return profile_select_native(profiles, profile_id)

    def upsert_prompt_enhancer_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        response = self._json(
            "/remote-prompt-enhancer/profiles",
            payload={"action": "upsert", "profile": profile},
        )
        from .._native import profile_response_native

        return profile_response_native(response, "upsert")

    def delete_prompt_enhancer_profile(self, profile_id: str) -> bool:
        response = self._json(
            "/remote-prompt-enhancer/profiles",
            payload={"action": "delete", "profile_id": profile_id},
        )
        from .._native import profile_response_native

        return profile_response_native(response, "delete")

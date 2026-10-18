"""Profile endpoints layered on the shared ComfyUI HTTP transport."""
from typing import Any

from ..core.transport import HttpComfyApi
from .profiles import PromptEnhancerProfile, validate_profile_payload


class PromptComfyApi(HttpComfyApi):
    def list_prompt_enhancer_profiles(self) -> list[dict[str, Any]]:
        response = self._json("/remote-prompt-enhancer/profiles")
        profiles = response.get("profiles")
        if set(response) != {"profiles"} or not isinstance(profiles, list):
            raise RuntimeError("Remote Prompt Enhancer profile list is invalid")
        return profiles

    def get_prompt_enhancer_profile(self, profile_id: str) -> PromptEnhancerProfile:
        profiles = self.list_prompt_enhancer_profiles()
        for payload in profiles:
            profile = validate_profile_payload(payload)
            if profile.id == profile_id:
                return profile
        raise ValueError(f"Remote Prompt Enhancer profile is not registered: {profile_id}")

    def upsert_prompt_enhancer_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        response = self._json(
            "/remote-prompt-enhancer/profiles",
            payload={"action": "upsert", "profile": profile},
        )
        stored = response.get("profile")
        if set(response) != {"profile"} or not isinstance(stored, dict):
            raise RuntimeError("Remote Prompt Enhancer profile upsert response is invalid")
        return stored

    def delete_prompt_enhancer_profile(self, profile_id: str) -> bool:
        response = self._json(
            "/remote-prompt-enhancer/profiles",
            payload={"action": "delete", "profile_id": profile_id},
        )
        deleted = response.get("deleted")
        if set(response) != {"deleted"} or not isinstance(deleted, bool):
            raise RuntimeError("Remote Prompt Enhancer profile delete response is invalid")
        return deleted


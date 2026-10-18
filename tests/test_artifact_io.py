from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import remoteworker.integration.artifact_io as io_conditioning_io


class io_FolderPathsFixture:
    def __init__(self, root: Path) -> None:
        self.input = root / "input"
        self.output = root / "output"
        self.input.mkdir()
        self.output.mkdir()

    def get_input_directory(self) -> str:
        return str(self.input)

    def get_output_directory(self) -> str:
        return str(self.output)

    def get_save_image_path(self, prefix: str, _output_dir: str) -> tuple[str, str, int, str, str]:
        relative = Path(prefix)
        subfolder = str(relative.parent) if str(relative.parent) != "." else ""
        return str(self.output / relative.parent), relative.name, 1, subfolder, ""


class ConditioningIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.paths = io_FolderPathsFixture(Path(self.temporary_directory.name))
        self.original_folder_paths = io_conditioning_io.folder_paths
        io_conditioning_io.folder_paths = self.paths

    def tearDown(self) -> None:
        io_conditioning_io.folder_paths = self.original_folder_paths
        self.temporary_directory.cleanup()

    def test_to_cpu_recursively_detaches_tensors(self) -> None:
        source = torch.tensor([1.0], requires_grad=True)

        result = io_conditioning_io._to_cpu({"nested": [source]})

        tensor = result["nested"][0]
        self.assertEqual("cpu", tensor.device.type)
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(torch.equal(tensor, source.detach().cpu()))

    def test_input_files_are_listed_and_take_precedence_over_output(self) -> None:
        (self.paths.input / "b.cond").write_bytes(b"input")
        (self.paths.input / "a.COND").write_bytes(b"input")
        (self.paths.input / "ignored.txt").write_bytes(b"ignored")
        (self.paths.output / "b.cond").write_bytes(b"output")

        self.assertEqual(["a.COND", "b.cond"], io_conditioning_io._list_input_conds())
        self.assertEqual(self.paths.input / "b.cond", Path(io_conditioning_io._resolve("b.cond")))

    def test_saved_conditioning_can_be_transferred_to_input_and_loaded(self) -> None:
        original = [[torch.tensor([1.0, 2.0]), {"pooled_output": torch.tensor([3.0])}]]

        saved = io_conditioning_io.SaveConditioning().save(original, "handoff/cond")

        output_path = self.paths.output / "handoff" / "cond_00001_.cond"
        self.assertTrue(output_path.is_file())
        self.assertEqual({"text": ["handoff/cond_00001_.cond"]}, saved["ui"])
        self.assertEqual(("handoff/cond_00001_.cond",), saved["result"])
        shutil.copy2(output_path, self.paths.input / output_path.name)

        loaded = io_conditioning_io.LoadConditioning().load(output_path.name)[0]

        self.assertTrue(torch.equal(loaded[0][0], original[0][0]))
        self.assertTrue(torch.equal(loaded[0][1]["pooled_output"], original[0][1]["pooled_output"]))

    def test_remote_latent_uses_torch_serialization_for_non_contiguous_values(
        self,
    ) -> None:
        original = {"samples": torch.tensor([[1.0, 2.0]]).t()}

        saved = io_conditioning_io.SaveRemoteLatent().save(original, "remote/latent")

        output_path = self.paths.output / "remote" / "latent_00001_.latent"
        restored = torch.load(output_path, map_location="cpu", weights_only=False)
        self.assertTrue(output_path.is_file())
        self.assertEqual(
            {
                "ui": {"latents": ["remote/latent_00001_.latent"]},
                "result": ("remote/latent_00001_.latent",),
            },
            saved,
        )
        self.assertTrue(torch.equal(restored["samples"], original["samples"]))

    def test_remote_latent_accepts_optional_dependency_for_cleanup_ordering(
        self,
    ) -> None:
        schema = io_conditioning_io.SaveRemoteLatent.INPUT_TYPES()

        self.assertEqual(("*",), schema["optional"]["dependency"])
        self.assertEqual(("STRING",), io_conditioning_io.SaveRemoteLatent.RETURN_TYPES)
        self.assertEqual(("artifact_path",), io_conditioning_io.SaveRemoteLatent.RETURN_NAMES)

    def test_remote_conditioning_uses_portable_request_scoped_artifact_path(
        self,
    ) -> None:
        saved = io_conditioning_io.SaveRemoteConditioning().save(
            [[torch.tensor([1.0])]],
            "remote-conditioning/request-1/positive",
        )

        self.assertEqual(
            {
                "ui": {"conditioning": ["remote-conditioning/request-1/positive_00001_.cond"]},
                "result": ("remote-conditioning/request-1/positive_00001_.cond",),
            },
            saved,
        )

    def test_remote_text_writes_one_utf8_artifact(self) -> None:
        saved = io_conditioning_io.SaveRemoteText().save("expanded prompt", "remote-conditioning/request-1/prompt")

        output_path = self.paths.output / "remote-conditioning" / "request-1" / "prompt_00001_.txt"
        self.assertTrue(output_path.is_file())
        self.assertEqual("expanded prompt", output_path.read_text(encoding="utf-8"))
        self.assertEqual({"ui": {"text": ["remote-conditioning/request-1/prompt_00001_.txt"]}}, saved)

    def test_missing_file_fails_before_load(self) -> None:
        filename = "missing.cond"

        self.assertIn(
            "conditioning 파일 없음",
            io_conditioning_io.LoadConditioning.VALIDATE_INPUTS(filename),
        )
        with self.assertRaises(FileNotFoundError):
            io_conditioning_io.LoadConditioning().load(filename)


import remoteworker.core.artifacts as artifact_remote_artifacts


class RemoteArtifactTests(unittest.TestCase):
    def test_recovers_declared_conditioning_and_latent_artifacts(self) -> None:
        latent = artifact_remote_artifacts.RemoteArtifactSpec("latent", "save_latent", "latents", ".latent")
        history = {
            "prompt-1": {
                "status": {"status_str": "success"},
                "outputs": {
                    "save": {"text": ["remote-conditioning/request-1/cond_00001_.cond"]},
                    "save_latent": {
                        "latents": [
                            {
                                "filename": "latent_00001_.latent",
                                "subfolder": "remote-conditioning/request-1",
                                "type": "output",
                            }
                        ]
                    },
                },
            },
        }

        self.assertEqual(
            {
                "conditioning": "remote-conditioning/request-1/cond_00001_.cond",
                "latent": "remote-conditioning/request-1/latent_00001_.latent",
            },
            artifact_remote_artifacts.extract_artifact_paths(
                history,
                "prompt-1",
                "request-1",
                (artifact_remote_artifacts.CONDITIONING_ARTIFACT, latent),
            ),
        )

    def test_rejects_locator_outside_request_scope(self) -> None:
        history = {
            "prompt-1": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"text": ["latents/other.cond"]}},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "outside its request directory"):
            artifact_remote_artifacts.extract_artifact_paths(
                history,
                "prompt-1",
                "request-1",
                (artifact_remote_artifacts.CONDITIONING_ARTIFACT,),
            )

    def test_recovers_one_utf8_prompt_artifact(self) -> None:
        history = {
            "prompt-1": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"text": ["remote-conditioning/request-1/prompt_00001_.txt"]}},
            },
        }
        self.assertEqual(
            {"prompt": "remote-conditioning/request-1/prompt_00001_.txt"},
            artifact_remote_artifacts.extract_artifact_paths(
                history,
                "prompt-1",
                "request-1",
                (artifact_remote_artifacts.PROMPT_ARTIFACT,),
            ),
        )

    def test_surfaces_remote_execution_detail(self) -> None:
        history = {
            "prompt-1": {
                "status": {
                    "status_str": "error",
                    "messages": [
                        [
                            "execution_error",
                            {
                                "node_type": "MiniMaxH3ReferenceToVideo",
                                "exception_type": "ValueError",
                                "exception_message": "missing VAE",
                            },
                        ]
                    ],
                },
                "outputs": {},
            },
        }
        with self.assertRaisesRegex(RuntimeError, r"MiniMaxH3ReferenceToVideo.*missing VAE"):
            artifact_remote_artifacts.extract_artifact_paths(
                history,
                "prompt-1",
                "request-1",
                (artifact_remote_artifacts.CONDITIONING_ARTIFACT,),
            )

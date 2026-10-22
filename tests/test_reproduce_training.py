"""No ML imports or training: test the preparation-stage command builder."""

import ast
import contextlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest

from scripts import reproduce_training as preview


def plan(model="sid", **updates):
    values = dict(model=model, data_root="/placeholder/data", sid_path="/placeholder/sids.npy",
                  output_root="/placeholder/output", seed=42, devices=3,
                  python_executable="python", project_root="/placeholder/repository")
    if model != "sid":
        values["teacher_checkpoint"] = "/placeholder/teacher.ckpt"
    values.update(updates)
    return preview.build_plan(**values)


class TrainingPreviewTests(unittest.TestCase):
    def test_exact_four_model_experiment_compositions(self):
        expected = {
            "sid": "experiment=[tiger_train_flat,tiger_train_sid]",
            "listwise": "experiment=[tiger_train_flat,tiger_train_click]",
            "pointwise": "experiment=[tiger_train_flat,tiger_train_click,tiger_train_click_pointwise]",
            "separate": "experiment=[tiger_train_flat,tiger_train_rerank_small]",
        }
        self.assertEqual(set(preview.MODELS), set(expected))
        for model, experiment in expected.items():
            with self.subTest(model=model):
                result = plan(model)
                argv = result["training_argv"]
                self.assertEqual(argv[:3], ["python", "-m", "src.train"])
                self.assertIn(experiment, argv)
                for setting in (
                    "num_hierarchies=4", "num_embeddings_per_hierarchy=4096",
                    "trainer.max_steps=120000", "optim.scheduler.num_training_steps=120000",
                    "trainer.devices=3", "trainer.strategy=ddp", "trainer.num_nodes=1",
                    "test=false", "train=true", "ckpt_path=null", "++should_skip_retry=true",
                    "data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false",
                    "data_loading.train_dataloader_config.dataloader.num_workers=1",
                    "data_loading.val_dataloader_config.dataloader.num_workers=2",
                    "data_loading.test_dataloader_config.dataloader.num_workers=3",
                ):
                    self.assertIn(setting, argv)
                self.assertFalse(result["execution_enabled"])
                self.assertFalse(result["formal_reproduction_verified"])

    def test_staged_initialization_and_objective_are_not_resume(self):
        sid = plan()["training_argv"]
        self.assertIn("model.load_only_weights=false", sid)
        self.assertFalse(any("ckpt_path=" in a and a != "ckpt_path=null" for a in sid))
        for model in ("listwise", "pointwise"):
            argv = plan(model)["training_argv"]
            self.assertIn('model.manual_ckpt_path="/placeholder/teacher.ckpt"', argv)
            self.assertIn("model.load_only_weights=true", argv)
            self.assertIn(f"model.rerank_objective={model}", argv)
            self.assertFalse(any(a.startswith("model.sid_teacher_ckpt_path=") for a in argv))
        separate = plan("separate")["training_argv"]
        self.assertIn('model.sid_teacher_ckpt_path="/placeholder/teacher.ckpt"', separate)
        self.assertIn("model.load_only_weights=false", separate)
        self.assertFalse(any(a.startswith("model.manual_ckpt_path=") for a in separate))

    def test_teacher_required_only_for_the_three_staged_models(self):
        for model in ("listwise", "pointwise", "separate"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "requires"):
                plan(model, teacher_checkpoint=None)
        with self.assertRaisesRegex(ValueError, "from scratch"):
            plan(teacher_checkpoint="/placeholder/unwanted.ckpt")

    def test_single_device_is_explicitly_not_equivalent_or_executable(self):
        result = plan(devices=1)
        self.assertIn("trainer.devices=1", result["training_argv"])
        self.assertIn("trainer.strategy=auto", result["training_argv"])
        self.assertEqual(result["batch"]["effective"], 128)
        self.assertEqual(result["batch"]["historical_effective"], 384)
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(any("not released" in gate for gate in result["execution_gates"]))
        self.assertTrue(any("fabricate scheduler" in gate for gate in result["execution_gates"]))

    def test_four_cpu_budget_and_single_thread_require_real_external_binding(self):
        for devices in (1, 3):
            result = plan(devices=devices)
            self.assertEqual(result["cpu_budget"], {"whole_job_ceiling": 4, "library_threads": 1})
            self.assertEqual(result["environment_preview"]["GRID_CPU_BUDGET"], "4")
            for name in preview.THREAD_VARIABLES:
                self.assertEqual(result["environment_preview"][name], "1")
            self.assertFalse(any(name.startswith("SLURM_") for name in result["environment_preview"]))

    def test_paths_with_shell_and_hydra_punctuation_remain_single_string_arguments(self):
        path = '/placeholder/a b,[x]=yes;$(not-a-command)/"quoted".npy'
        result = plan(sid_path=path)
        quoted = next(a.split("=", 1)[1] for a in result["training_argv"] if a.startswith("semantic_id_path="))
        self.assertEqual(json.loads(quoted), path)
        self.assertEqual(shlex.split(result["training_command_preview"]), result["training_argv"])

    def test_missing_placeholder_inputs_are_not_opened_or_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.iterdir())
            result = plan(data_root=str(root / "missing_data"), sid_path=str(root / "missing_sid.npy"),
                          output_root=str(root / "new_output"), project_root=root)
            self.assertEqual(result["root_marker"], str(root / ".project-root"))
            self.assertEqual(list(root.iterdir()), before)

    def test_preflight_is_separate_and_teacher_identity_is_explicit(self):
        sid = plan()["preflight_argv"]
        self.assertEqual(sid[:3], ["python", "-m", "scripts.verify_taobao_training_data"])
        self.assertIn("history-last-user-v1", sid)
        self.assertNotIn("--initialization-checkpoint", sid)
        staged = plan("listwise")["preflight_argv"]
        index = staged.index("--initialization-checkpoint")
        self.assertEqual(staged[index + 1], "/placeholder/teacher.ckpt")

    def test_preflight_binds_distributed_core_sources_and_model_configs(self):
        expected_core = {
            "scripts/reproduce_training.py", "scripts/verify_taobao_training_data.py",
            "src/train.py", "src/runtime_cpu.py", "src/checkpoint_resume.py",
            "src/data/loading/components/iterators.py",
            "src/data/loading/components/dataloading.py",
            "src/data/loading/components/interfaces.py",
            "src/models/modules/semantic_id/tiger_generation_model.py",
        }
        separate_source = "src/models/modules/semantic_id/separate_rerank_model.py"
        for model in preview.MODELS:
            with self.subTest(model=model):
                result = plan(model)
                argv = result["preflight_argv"]
                sources = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--source-file"]
                expected = expected_core | set(result["source_experiments"])
                if model == "separate":
                    expected.add(separate_source)
                self.assertEqual(set(sources), expected)
                self.assertEqual(len(sources), len(expected))
                self.assertEqual(sources, result["preflight_source_files"])
                for source in sources:
                    self.assertFalse(Path(source).is_absolute())
                    self.assertNotIn("..", Path(source).parts)
                    self.assertTrue((preview.ROOT / source).is_file(), source)
                    self.assertFalse(source.endswith(".sh"), source)
                self.assertTrue(any("eventual launcher, environment lock" in gate
                                    for gate in result["execution_gates"]))

    def test_invalid_model_seed_devices_and_paths_are_rejected(self):
        for key, value in (("model", "liger"), ("model", "sim"), ("seed", -1),
                           ("seed", True), ("devices", 2), ("devices", True),
                           ("data_root", ""), ("data_root", None), ("sid_path", "a\nb"),
                           ("output_root", "${environment:unsafe}")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                plan(**{key: value})

    def test_cli_default_and_explicit_dry_run_only_print_json(self):
        args = ["--model", "sid", "--data-root", "/placeholder/data", "--sid-path",
                "/placeholder/sid.npy", "--output-root", "/placeholder/output"]
        for suffix in ([], ["--dry-run"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                preview.main(args + suffix)
            result = json.loads(output.getvalue())
            self.assertEqual(result["mode"], "dry_run_only")
            self.assertFalse(result["execution_enabled"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            preview.main(args + ["--execute"])
        self.assertEqual(error.exception.code, 2)

    def test_module_imports_only_stdlib_and_has_no_process_launch_api(self):
        source = Path(preview.__file__).read_text()
        tree = ast.parse(source)
        allowed = {"argparse", "json", "pathlib", "shlex", "sys"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue({alias.name for alias in node.names} <= allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
        for forbidden in ("subprocess", "torchrun", "import torch", "import tensorflow", "sbatch("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

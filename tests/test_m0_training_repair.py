"""M0 repair tests for KL alignment, grad flow, and smoke training."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from where_to_bake.config import load_config
from where_to_bake.models import create_teacher_student_pair
from where_to_bake.models.wrapper import _require_hf_stack, load_tokenizer
from where_to_bake.train.losses import align_student_teacher_logits, compute_token_kl
from where_to_bake.train.trainer import _build_dataloader, _build_dataset, run_experiment

HAS_HF_STACK = importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("peft") is not None


class M0TrainingRepairTest(unittest.TestCase):
    def test_align_student_teacher_logits_uses_shifted_prediction_positions(self) -> None:
        import torch

        student_logits = torch.tensor(
            [
                [
                    [10.0, 0.0],
                    [20.0, 0.0],
                    [30.0, 0.0],
                    [40.0, 0.0],
                ]
            ]
        )
        teacher_logits = torch.tensor(
            [
                [
                    [11.0, 0.0],
                    [21.0, 0.0],
                    [31.0, 0.0],
                    [41.0, 0.0],
                ]
            ]
        )
        response_mask = torch.tensor([[0, 0, 1, 1]])

        aligned_student, aligned_teacher = align_student_teacher_logits(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_response_mask=response_mask,
            teacher_response_mask=response_mask,
        )

        self.assertEqual(aligned_student.shape, (2, 2))
        self.assertTrue(torch.equal(aligned_student[:, 0], torch.tensor([20.0, 30.0])))
        self.assertTrue(torch.equal(aligned_teacher[:, 0], torch.tensor([21.0, 31.0])))

    @unittest.skipUnless(HAS_HF_STACK, "transformers+peft are required for HF smoke tests")
    def test_teacher_is_detached_but_student_and_loss_keep_grad_graph(self) -> None:
        torch, _auto_model, auto_tokenizer, _peft = _require_hf_stack()
        config = load_config("configs/baselines/promptbake_kl.yaml")
        tokenizer = load_tokenizer(config, auto_tokenizer)
        dataset = _build_dataset(
            tokenizer=tokenizer,
            config=config,
            split_path=config["data"]["train_path"],
            paraphrase_split=config["train"].get("train_paraphrase_split", "seen"),
        )
        batch = next(iter(_build_dataloader(dataset, batch_size=2)))
        teacher_model, student_model, _tokenizer, _runtime_torch = create_teacher_student_pair(
            config=config,
            device=config["run"]["device"],
            target_modules=list(config["lora"]["target_modules"]),
        )
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

        teacher_inputs = {
            "input_ids": batch["teacher_input_ids"].to(config["run"]["device"]),
            "attention_mask": batch["teacher_attention_mask"].to(config["run"]["device"]),
        }
        student_inputs = {
            "input_ids": batch["student_input_ids"].to(config["run"]["device"]),
            "attention_mask": batch["student_attention_mask"].to(config["run"]["device"]),
        }

        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
        student_outputs = student_model(**student_inputs)
        loss = config["loss"]["kl_weight"] * compute_token_kl(
            student_outputs.logits,
            teacher_outputs.logits,
            batch["student_response_mask"].to(config["run"]["device"]),
            batch["teacher_response_mask"].to(config["run"]["device"]),
            temperature=config["loss"]["temperature"],
        )

        self.assertFalse(teacher_outputs.logits.requires_grad)
        self.assertTrue(student_outputs.logits.requires_grad)
        self.assertTrue(loss.requires_grad)

        loss.backward()
        lora_grad_norm = 0.0
        for name, parameter in student_model.named_parameters():
            if parameter.requires_grad and "lora_" in name and parameter.grad is not None:
                lora_grad_norm += parameter.grad.detach().float().norm(p=2).item()
        self.assertGreater(lora_grad_norm, 0.0)

    @unittest.skipUnless(HAS_HF_STACK, "transformers+peft are required for HF smoke tests")
    def test_promptbake_smoke_train_reports_nonzero_lora_grad_norm(self) -> None:
        config = load_config("configs/baselines/promptbake_kl.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output"]["output_dir"] = str(Path(temp_dir) / "promptbake_kl_smoke")
            config["eval"]["max_eval_batches"] = 1
            config["eval"]["max_new_tokens"] = 4
            result = run_experiment(config)

            self.assertIsNotNone(result["notes"]["last_train_loss"])
            self.assertGreater(result["notes"]["max_trainable_grad_norm"], 0.0)
            self.assertGreater(result["notes"]["nonzero_grad_steps"], 0)
            self.assertEqual(result["loss_weights"]["kl"]["status"], "used")
            self.assertEqual(result["loss_weights"]["delta"]["status"], "not_used")
            self.assertEqual(result["loss_weights"]["preserve"]["status"], "not_used")
            self.assertTrue((Path(config["output"]["output_dir"]) / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

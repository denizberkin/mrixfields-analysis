import sys
import tempfile
import unittest
from pathlib import Path

from scripts.generate_challenge_submission import Mapping, RunSpec, run_custom_inference


class CustomInferenceTest(unittest.TestCase):
    def test_runs_command_with_mapping_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = Mapping("T1W", "0.1T", "1.5T", root / "weights.pt", None)
            code = (
                "from pathlib import Path; p=Path(r'{output_dir}'); "
                "[(p / n).touch() for n in "
                "('P_T1W_0.1T_0001.nii.gz','P_T1W_0.1T_0002.nii.gz','P_T1W_0.1T_0003.nii.gz')]"
            )
            spec = RunSpec(
                name="custom",
                task="task2",
                method="custom",
                mode="run",
                epoch_tag="final",
                seed=0,
                device="cpu",
                inference_command=(sys.executable, "-c", code),
                mappings=[mapping],
                prediction_root=root / "predictions",
                segmentation_root=root / "segmentations",
                submission_root=root / "submission",
                synthseg_dir=root / "SynthSeg",
            )

            run_custom_inference(spec, root / "data", {}, overwrite=False, dry_run=False)

            self.assertEqual(
                {path.name for path in (root / "predictions").rglob("*.nii.gz")},
                {
                    "P_T1W_0.1T_0001.nii.gz",
                    "P_T1W_0.1T_0002.nii.gz",
                    "P_T1W_0.1T_0003.nii.gz",
                },
            )


if __name__ == "__main__":
    unittest.main()

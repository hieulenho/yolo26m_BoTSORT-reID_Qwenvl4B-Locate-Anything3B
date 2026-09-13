from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from football_tracking.task_pipeline.config import load_task_pipeline_config


def test_public_default_has_no_extra_model_or_private_checkpoint() -> None:
    task = load_task_pipeline_config('configs/tasks/traffic_realtime.yaml')
    assert task.detector.checkpoint == 'yolo26n.pt'
    assert not task.detector.supplemental_detectors
    assert task.semantic.model_id == 'Qwen/Qwen3-VL-4B-Instruct'
    assert task.semantic.quantization == '8bit'
    assert task.semantic.max_evidence_images == 2


def test_gt_runner_does_not_drop_reference_frames() -> None:
    script = Path('scripts/benchmarks/run_task_gt_benchmark.ps1').read_text(encoding='utf-8')
    assert 'DisableFrameDropping = $true' in script
    assert 'SynchronousVideoWrite = $true' in script


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell process quoting')
def test_worker_arguments_preserve_spaces_quotes_and_empty_values(tmp_path: Path) -> None:
    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    helper = Path('scripts/runtime/process_arguments.ps1').resolve()
    expected = ['path with spaces', 'quote"inside', 'trailing \\', '']
    values = ['-c', 'import json,sys; print(json.dumps(sys.argv[1:]))', *expected]
    output = tmp_path / 'child output.json'
    script = (
        f'. {quote(str(helper))}\n'
        f'$values = @({", ".join(quote(v) for v in values)})\n'
        f'$p = Start-Process -FilePath {quote(sys.executable)} '
        '-ArgumentList (ConvertTo-WindowsArguments $values) -WindowStyle Hidden '
        f'-Wait -PassThru -RedirectStandardOutput {quote(str(output))}\n'
        'exit $p.ExitCode'
    )
    subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True, timeout=30)
    assert json.loads(output.read_text(encoding='utf-8')) == expected


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell parser')
def test_all_powershell_scripts_parse() -> None:
    script = '''
    $ErrorActionPreference = 'Stop'
    $failed = 0
    Get-ChildItem scripts -Recurse -Filter *.ps1 | ForEach-Object {
        $tokens = $null
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile(
            $_.FullName, [ref]$tokens, [ref]$errors)
        if ($errors.Count) { $errors | Out-String | Write-Host; $failed += 1 }
    }
    exit $failed
    '''
    subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True, timeout=30)

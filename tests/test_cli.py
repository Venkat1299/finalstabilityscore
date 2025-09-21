import subprocess

def test_cli_help():
    result = subprocess.run(["python3", "-m", "finalstabilityscore.cli", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"usage" in result.stdout

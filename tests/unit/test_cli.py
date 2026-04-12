"""CLI test suite."""

import sys

import pytest

from bda_svc import cli


def test_get_args(mocker, capsys):
    """Test argparse wrapper.

    Args:
        mocker: pytest-mock fixture
        capsys: pytest fixture to retrieve function STDOUT|STDIN|STDERR
    """
    # Test help functionality
    for arg_help in ("-h", "--help"):
        mocker.patch.object(sys, "argv", ["bda-svc", arg_help])

        # `bda-svc -h` always exits with "sys.exit(0)", raising a SystemExit exception
        with pytest.raises(SystemExit) as exec_help:
            cli.get_args()

        # argparse should return 0 when "-h" or "--help" used
        assert exec_help.value.code == 0

        captured = capsys.readouterr()
        assert "usage" in captured.out

    # Test input argument functionality
    tmp_filename = "image42.png"

    for arg_input in ("-i", "--input"):
        mocker.patch.object(sys, "argv", ["bda-svc", arg_input, tmp_filename])

        args = cli.get_args()

        # Check if argparse.Namespace object contains our argument label
        # and if it is set to the filename we provided
        assert hasattr(args, "input") and args.input == tmp_filename

        # Test with missing filename
        mocker.patch.object(sys, "argv", ["bda-svc", arg_input])

        with pytest.raises(SystemExit) as exec_missing_input:
            cli.get_args()

        assert exec_missing_input.value.code == 2

        captured = capsys.readouterr()

    # Test output argument functionality
    tmp_output_folder = "/path/to/nowhere"

    for arg_output in ("-o", "--output"):
        mocker.patch.object(sys, "argv", ["bda-svc", arg_output, tmp_output_folder])

        args = cli.get_args()

        # Check if argparse.Namespace object contains our argument label
        # and if it is set to the folder we provided
        assert hasattr(args, "output") and args.output == tmp_output_folder

        # Test with missing output folder
        mocker.patch.object(sys, "argv", ["bda-svc", arg_output])

        with pytest.raises(SystemExit) as exec_missing_output:
            cli.get_args()

        assert exec_missing_output.value.code == 2

        captured = capsys.readouterr()

    # Test debug export flag
    mocker.patch.object(sys, "argv", ["bda-svc", "--debug-export-images"])
    args = cli.get_args()
    assert hasattr(args, "debug_export_images")
    assert args.debug_export_images is True

    mocker.patch.object(sys, "argv", ["bda-svc"])
    args = cli.get_args()
    assert hasattr(args, "debug_export_images")
    assert args.debug_export_images is False

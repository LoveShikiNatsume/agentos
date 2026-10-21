from scripts.record_demo import parse_args


def test_record_demo_preserves_full_hd_defaults():
    args = parse_args([])
    assert (args.width, args.height) == (1920, 1080)
    assert args.filename == "demo_raw.webm"


def test_record_demo_accepts_explicit_qhd_output():
    args = parse_args([
        "--width", "2560",
        "--height", "1440",
        "--storyboard", "runs/demo_video/storyboard_qhd_refined.json",
        "--filename", "demo_qhd_refined.webm",
    ])
    assert (args.width, args.height) == (2560, 1440)
    assert args.filename == "demo_qhd_refined.webm"

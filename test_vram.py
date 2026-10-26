from vram import estimate_vram, gpus_that_fit, kv_cache_gb, weight_gb


def test_weights():
    # 70 x 2 = 140 GB in BF16, 70 x 0.5 = 35 GB in INT4.
    assert weight_gb(70, 2) == 140
    assert weight_gb(70, 0.5) == 35


def test_kv_cache():
    # 2 x 80 x 1024 x 8192 x 1 x 2 = 2,684,354,560 bytes.
    assert abs(kv_cache_gb(80, 1024, 8192, 1) - 2.68435456) < 1e-9


def test_total_70b_bf16():
    # (140 + 2.68435456) x 1.1 = 156.952790016 GB.
    result = estimate_vram(70, "bf16", 80, 1024, 8192, 1, 10)
    assert abs(result["total_gb"] - 156.952790016) < 1e-9


def test_gpus_that_fit():
    gpus = [("A 80", 80), ("B 141", 141), ("C 192", 192), ("D 24", 24), ("E 16", 16)]
    names = [(name, count) for name, _, count in gpus_that_fit(157, gpus)]
    assert names == [("C 192", 1), ("A 80", 2), ("B 141", 2), ("D 24", 8)]

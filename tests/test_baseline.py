from agent_wars_devfest.tools.baseline import END_DAY, run_fcfs


def test_fcfs_is_deterministic():
    assert run_fcfs("act1") == run_fcfs("act1")


def test_fcfs_runs_every_act_to_the_end():
    for act in ("act1", "act2", "act3"):
        s = run_fcfs(act)
        assert s["day"] == END_DAY
        assert 0 < s["fulfillment_rate"] <= 1
        assert s["total_cost"] > 0


def test_fcfs_wastes_the_cancelled_load_in_act3():
    # No reallocation: the truck already heading to C3 delivers into a cancelled order.
    assert run_fcfs("act3")["wasted_t"] >= 20

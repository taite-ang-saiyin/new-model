from action_engine.envs.character_env import MultiObjectiveCharacterEnv


def test_character_env_emits_narrative_metrics():
    env = MultiObjectiveCharacterEnv(state_dim=4, memory_len=2, max_steps=4)
    obs, info = env.reset(options={"story_context": {"agenda_target": 0.8}})

    assert "narrative" in obs
    assert obs["narrative"].shape[0] == env.storyline_dim
    assert "story_context" in info

    env.inject_story_metrics({"agenda_progress": 0.9, "tension": 0.2})
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert obs["narrative"].shape[0] == env.storyline_dim
    assert reward.shape[0] == 3
    assert isinstance(info["narrative_metrics"], dict)

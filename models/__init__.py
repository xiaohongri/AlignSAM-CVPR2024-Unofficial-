from models.implicit_agent import ImplicitAgent
from models.explicit_agent import ExplicitAgent


def make_agent(agent_cfg, envs):
    if agent_cfg['type'] == "implicit":
        return ImplicitAgent(envs, agent_cfg)
    elif agent_cfg['type'] == "explicit":
        return ExplicitAgent(envs, agent_cfg)
    else:
        raise ValueError(f"Unknown agent type {agent_cfg['type']}")
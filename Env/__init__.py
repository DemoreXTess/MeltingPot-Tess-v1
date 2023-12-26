from gym.envs.registration import register

register("TessEnv-v1",entry_point="Env.tess_env:TessEnv")
register("TessEnv-v2",entry_point="Env.tess_env-v2:TessEnv")
register("TessEnv-v3",entry_point="Env.tess_env-v3:TessEnv")
register("TessEnv-v4",entry_point="Env.tess_env-v4-multi-task:TessEnv")
register("TerritorySpecific-v1",entry_point="Env.territory_specific:TessEnv")
"""Fuzzing support modules for LSGEmu."""

__all__ = [
    "ByteMutator",
    "CampaignIteration",
    "CrashDetector",
    "CrashDetectorConfig",
    "CrashReport",
    "CrashSignal",
    "CrashStore",
    "CorpusStore",
    "FuzzCampaign",
    "FuzzCampaignConfig",
    "FuzzProfile",
    "LSGEmuRunConfig",
    "LSGEmuSeedRunResult",
    "LSGEmuSeedRunner",
    "MutationConfig",
    "RuntimeCrashEvent",
    "RuntimeCrashMonitor",
    "build_security_report",
    "classify_security_observation",
    "collect_campaign_findings",
    "extract_attack_surface",
    "security_report_markdown",
    "SeedRecord",
    "SupervisedRunResult",
    "detect_many_from_lsgemu_report",
    "build_profile_from_report",
    "load_campaign_config",
    "load_profile",
    "run_supervised",
    "write_profile",
]


def __getattr__(name):
    if name in {
        "CrashDetector",
        "CrashDetectorConfig",
        "CrashReport",
        "CrashSignal",
        "CrashStore",
        "detect_many_from_lsgemu_report",
    }:
        from . import crash_detector

        return getattr(crash_detector, name)
    if name in {"RuntimeCrashEvent", "RuntimeCrashMonitor"}:
        from . import runtime_crash_monitor

        return getattr(runtime_crash_monitor, name)
    if name in {"SupervisedRunResult", "run_supervised"}:
        from . import subprocess_runner

        return getattr(subprocess_runner, name)
    if name in {"CorpusStore", "SeedRecord"}:
        from . import corpus

        return getattr(corpus, name)
    if name in {"ByteMutator", "MutationConfig"}:
        from . import mutator

        return getattr(mutator, name)
    if name in {"LSGEmuRunConfig", "LSGEmuSeedRunResult", "LSGEmuSeedRunner"}:
        from . import lsgemu_runner

        return getattr(lsgemu_runner, name)
    if name in {"CampaignIteration", "FuzzCampaign", "FuzzCampaignConfig", "load_campaign_config"}:
        from . import campaign

        return getattr(campaign, name)
    if name in {"FuzzProfile", "build_profile_from_report", "load_profile", "write_profile"}:
        from . import profile

        return getattr(profile, name)
    if name in {
        "build_security_report",
        "classify_security_observation",
        "collect_campaign_findings",
        "extract_attack_surface",
        "security_report_markdown",
    }:
        from . import security_analysis

        return getattr(security_analysis, name)
    raise AttributeError(name)

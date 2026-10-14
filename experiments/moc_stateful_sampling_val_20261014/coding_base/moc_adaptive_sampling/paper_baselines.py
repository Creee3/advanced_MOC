"""Thin delegation to MOC's published baselines; no replacement agent prompts."""

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
import importlib.util
import inspect
import subprocess
import sys


MOC_COMMIT = "9c67c92507570704a7df73e452552a3f49e83897"
PAPER_BASELINES = ("single_llm", "vanilla_mas", "moc")


@dataclass(frozen=True)
class PaperRecipe:
    # AQuA is the first supported domain, not an invented benchmark/baseline.
    domain: str = "aqua"
    llm_name: str = "gemma2:27b"
    num_agents: int = 7
    edge_density: float = 0.5
    random_dag_seed: int = 42
    use_cot: bool = False
    neighbor_hops: int = 2
    ism_r: float = 1.0
    ism_epsilon: float = 0.1
    ism_kppa: int = 45

    def __post_init__(self):
        if self.domain not in ("aqua", "gsm8k", "svamp"):
            raise NotImplementedError("candidate adapter v0 supports only MathSolver domains")
        if self.num_agents != 7 or self.edge_density not in (0.3, 0.5, 0.7, 1.0):
            raise ValueError("main-table recipe uses 7 agents and paper density values")
        if self.neighbor_hops < 1:
            raise ValueError("neighbor_hops must be positive")


@dataclass
class AuthorAPI:
    Graph: object
    get_kwargs: object
    process_single_question: object
    LLMRegistry: object
    PromptSetRegistry: object


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_author_api(author_root):
    """Lazy; importing this extension alone does not load models or the author."""
    root = Path(author_root).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != MOC_COMMIT:
        raise ValueError(f"expected pinned MOC {MOC_COMMIT}, found {commit}")
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise ValueError("author tracked files are modified; record/resolve changes before a baseline run")
    # Avoid importing a previously loaded 'src' from another research checkout.
    if "src" in sys.modules:
        package_paths = list(getattr(sys.modules["src"], "__path__", ()))
        if not any(Path(p).resolve() == root / "src" for p in package_paths):
            raise RuntimeError("another checkout's src package is loaded; use a fresh process")
    sys.path.insert(0, str(root))
    from src.graph.graph import Graph
    from src.llm.llm_registry import LLMRegistry
    from src.prompt.prompt_set_registry import PromptSetRegistry
    experiment = _load_file(root / "experiments/run_experiment.py", "_pinned_moc_experiment")
    direct = _load_file(root / "experiments/run_direct.py", "_pinned_moc_direct")
    return AuthorAPI(Graph, experiment.get_kwargs, direct.process_single_question, LLMRegistry, PromptSetRegistry)


def make_graph(api, recipe, arm, *, roles=None):
    if arm not in ("vanilla_mas", "moc", "moc_without_phi", "adaptive_sampling"):
        raise ValueError("Single LLM delegates to run_direct, not an edgeless MAS")
    topology = api.get_kwargs("Random", recipe.num_agents, recipe.edge_density, recipe.random_dag_seed)
    if roles is not None:
        if len(roles) != recipe.num_agents:
            raise ValueError("role snapshot has the wrong node count")
        topology["node_kwargs"] = [{"role": role} for role in roles]
    graph = api.Graph(domain=recipe.domain, llm_name=recipe.llm_name,
                      agent_names=["MathSolver"] * recipe.num_agents, decision_method="FinalRefer",
                      use_neighbor_summary=arm != "vanilla_mas",
                      neighbor_hops=recipe.neighbor_hops if arm != "vanilla_mas" else 1,
                      ism_r=recipe.ism_r, ism_epsilon=recipe.ism_epsilon, ism_kppa=recipe.ism_kppa,
                      use_cot=recipe.use_cot, **topology)
    if arm == "moc_without_phi":
        # Figure 3(b) ablation: keep original ancestor collection/serialization,
        # remove only the iterative merging operator, including its LLM calls.
        async def without_phi(_graph, neighbors, **kwargs):
            return "\n\n".join(f"Agent {id},role is {role},output is:\n{message}"
                               for hop, id, role, message in neighbors)
        graph.iterative_semantic_merging_with_clustering = MethodType(without_phi, graph)
    return graph


def make_comparison_graphs(api, recipe, sampler, encode, **adapter_kwargs):
    """Author roles assigned once, then frozen identically across compared arms."""
    from .adapter import MOCSamplingAdapter
    vanilla = make_graph(api, recipe, "vanilla_mas")
    roles = [node.role for node in vanilla.nodes.values()]
    graphs = {"vanilla_mas": vanilla,
              "moc": make_graph(api, recipe, "moc", roles=roles),
              "adaptive_sampling": make_graph(api, recipe, "adaptive_sampling", roles=roles)}
    adapter = MOCSamplingAdapter(graphs["adaptive_sampling"], sampler, encode, **adapter_kwargs)
    return graphs, adapter, roles


async def run_single_llm(api, recipe, question):
    prompt = api.PromptSetRegistry.get(recipe.domain)
    # Pinned AQUAPromptSet declares a staticmethod with (self, answer).
    # Bind only that documented signature bug; original inference stays intact.
    if list(inspect.signature(prompt.postprocess_answer).parameters) == ["self", "answer"]:
        class ParserBinding:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def postprocess_answer(self, answer):
                return self.original.postprocess_answer(self.original, answer)
        prompt = ParserBinding(prompt)
    return await api.process_single_question(api.LLMRegistry.get(recipe.llm_name),
                                             prompt,
                                             question, use_cot=recipe.use_cot)

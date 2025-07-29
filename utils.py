
import json 
from time import time
from tqdm.auto import tqdm
from typing import Mapping, TypedDict, Any, Union, Optional
import pandas as pd
import re
import json
from pathlib import Path
import torch

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
    
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import  CallbackManagerForRetrieverRun
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument

from qdrant_client import QdrantClient, models
from pydantic import Field

import ranx
from ranx.data_structures.frozenset_dict import FrozensetDict
from collections import defaultdict
import numpy as np
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# local
from custom_report import Report

class Results(TypedDict):
    documents: list[Document]
    time_seconds: float

class ResultsTime(TypedDict):
    results: Results 
    total_time_seconds: float

class Run(TypedDict):
    results: Mapping[str, Mapping[str, int]]
    time_seconds: Mapping[str, float]
    total_time_seconds: float


class QdrantMRLRetriever(BaseRetriever):
    """Custom retriever for Qdrant with three-stage Matryoshka embeddings"""
    
    # fields that Pydantic should recognize
    client: QdrantClient = Field(...)
    collection_name: str = Field(...)
    model_name: str = Field(...)
    device: str = Field(default="cpu")
    mrl_dims: list[int] = Field(default=[256, 512, 768])
    content_payload_key: str = Field(default="page_content")
    metadata_payload_key: str = Field(default="metadata")
    vector_names: list[str] = Field(default=["small-embeddings", "medium-embeddings", "large-embeddings"])
    limits: list[int] = Field(default=[300, 200, 100])
    query_prefix: str = Field(default="")
    
    #  computed fields not stored in Pydantic
    sm_model: Any = Field(default=None, exclude=True)
    md_model: Any = Field(default=None, exclude=True)
    lg_model: Any = Field(default=None, exclude=True)
    sm_vector_name: str = Field(default="", exclude=True)
    md_vector_name: str = Field(default="", exclude=True)
    lg_vector_name: str = Field(default="", exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize computed fields after Pydantic initialization
        self.sm_vector_name, self.md_vector_name, self.lg_vector_name = self.vector_names
        
        sm_dim, md_dim, lg_dim = self.mrl_dims
       
        self.sm_model = SentenceTransformer(self.model_name, truncate_dim=sm_dim, device=self.device)
        self.md_model = SentenceTransformer(self.model_name, truncate_dim=md_dim, device=self.device)
        self.lg_model = SentenceTransformer(self.model_name, truncate_dim=lg_dim, device=self.device)
    

    def _get_relevant_documents(self, query, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        query = self.query_prefix + query
        sm_lim, md_lim, lg_lim = self.limits 

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=models.Prefetch(
                    prefetch=models.Prefetch(
                        query=self.sm_model.encode(query),
                        using=self.sm_vector_name,
                        limit=sm_lim,
                    ),
                    query=self.md_model.encode(query),
                    using=self.md_vector_name,
                    limit=md_lim,
                ),
                query=self.lg_model.encode(query),
                using=self.lg_vector_name,
                limit=lg_lim
            ).points
            
            
            results_docs = [  
                self._document_from_point(scored_point=res)
                for res in results
            ]
            
            return results_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def _document_from_point(self, scored_point: Any) -> Document:
        metadata = scored_point.payload.get(self.metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = self.collection_name
        metadata["_score"] = scored_point.score
        
        return Document(
            page_content=scored_point.payload.get(self.content_payload_key, ""),
            metadata=metadata,
        )
    

class MedCPTEmbeddings(Embeddings):
    """Custom Embeddings for MedCPT Article and Query Encoder """

    def __init__(self, model_name: str, query_model_name: str, device:str):
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

        self.query_model = AutoModel.from_pretrained(query_model_name,  device_map="auto")
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name, device_map="auto")

        self.device = device

    def embed_documents(self, texts: list[str], tokenizer: Optional[AutoTokenizer] = None, model: Optional[AutoModel] = None, max_length: int=512) -> list[list[float]]:
        if tokenizer is None:
            tokenizer = self.tokenizer
        if model is None: 
            model = self.model
        
        def embed_doc(doc: str):
            with torch.no_grad():
                encoded = tokenizer(
                    doc, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=max_length,
                ).to(self.device)
            
            # encode the queries (use the [CLS] last hidden states as the representations)
                embeds = model(**encoded).last_hidden_state[:, 0, :]
                return embeds[0].tolist()
    
        return [embed_doc(doc) for doc in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text], 
                                    tokenizer=self.query_tokenizer, 
                                    model=self.query_model, 
                                    max_length=62)[0]



def list_to_str(list: list) -> str:
    string = ', '.join(list)
    return string

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(' +',' ',text)
    return text.strip()


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_corpus_documents_huggingface(repo_id: str, 
                                     filename: str, 
                                     cache_dir: Union[str, Path, None] = None, 
                                     local_dir: Union[str, Path, None] = None) -> list[Document]:
    df_corpus = pd.read_parquet(hf_hub_download(repo_id=repo_id, 
                                                filename=filename, 
                                                repo_type="dataset",
                                                cache_dir=cache_dir,
                                                local_dir=local_dir))

    df_corpus["authors"] = df_corpus["authors"].apply(list_to_str)
    df_corpus["keywords"] = df_corpus["keywords"].apply(list_to_str)
    df_corpus["publish_type"] = df_corpus["publish_type"].apply(list_to_str)
    df_corpus["passage"] = df_corpus["passage"].apply(clean_text)

    assert df_corpus["id"].duplicated().any() == False, "ID column contains duplicates. IDs should be unique."

    corpus_loader = DataFrameLoader(df_corpus, page_content_column="passage")
    corpus_docs = corpus_loader.load()

    return corpus_docs


def get_queries_huggingface(repo_id: str, 
                            filename: str, 
                            cache_dir: Union[str, Path, None] = None, 
                            local_dir: Union[str, Path, None] = None) -> dict[str, str]:
    
    df_test = pd.read_parquet(hf_hub_download(repo_id=repo_id, 
                                                filename=filename, 
                                                repo_type="dataset",
                                                cache_dir=cache_dir,
                                                local_dir=local_dir))
    df_test["id"] = df_test["id"].astype(str) 
    
    queries = dict(zip(df_test["id"], df_test["question"])) 

    return queries 



def get_qrels_huggingface(repo_id: str, 
                          filename: str, 
                          cache_dir: Union[str, Path, None] = None, 
                          local_dir: Union[str, Path, None] = None) -> dict[str, list[int]]:
    
    df_test = pd.read_parquet(hf_hub_download(repo_id=repo_id, 
                                                filename=filename, 
                                                repo_type="dataset",
                                                cache_dir=cache_dir,
                                                local_dir=local_dir))
    df_test["id"] = df_test["id"].astype(str) 
    
    def has_duplicates_in_list(lst):
        return len(lst) != len(set(lst))

    assert df_test["relevant_passage_ids"].apply(has_duplicates_in_list).any() == False, "List of relevant passage IDs contains duplicates."

    df_test["relevant_passage_ids"] = df_test["relevant_passage_ids"].apply(lambda x: x.tolist())

    qrels = dict(zip(df_test["id"], df_test["relevant_passage_ids"]))    

    return qrels


def get_eval_qrels_huggingface(repo_id: str,
                               filename: str, 
                               cache_dir: Union[str, Path, None] = None, 
                               local_dir: Union[str, Path, None] = None ) -> dict[str, dict[str, int]]:
    
    df_test = pd.read_parquet(hf_hub_download(repo_id=repo_id, 
                                                filename=filename, 
                                                repo_type="dataset",
                                                cache_dir=cache_dir,
                                                local_dir=local_dir))
    
    df_test["id"] = df_test["id"].astype(str) 
    
    def has_duplicates_in_list(lst):
        return len(lst) != len(set(lst))

    assert df_test["relevant_passage_ids"].apply(has_duplicates_in_list).any() == False, "List of relevant passage IDs contains duplicates."

    df_test["relevant_passage_ids"] = df_test["relevant_passage_ids"].apply(lambda x: x.tolist())

    qrels_dict = {}

    for q_id, rel_docs in zip(df_test["id"], df_test["relevant_passage_ids"]):
        qrels_dict[q_id] = {str(doc_id): 1 for doc_id in rel_docs}

    return qrels_dict


def retrieve(retriever: VectorStoreRetriever, queries_dict: Mapping[str, str]) -> ResultsTime:
    results = {
        "results": {}
    }

    total_start_time = time()

    for q_id, query in tqdm(queries_dict.items()):
        start_time = time()
        rets = retriever.invoke(query)
        end_time = time()
        results["results"][q_id] = {
            "documents": rets,
            "time_seconds": end_time - start_time
        }

    total_end_time = time()

    results["total_time_seconds"] = total_end_time - total_start_time

    return results


def format_run(results: ResultsTime, qrels: Mapping[str, list[int]]) -> Run:
    run = {
        "results": {},
        "time_seconds": {},
        "total_time_seconds": results["total_time_seconds"]
    }

    for q_id, res in results["results"].items():
        rel_docs = qrels[q_id]
        run["results"][q_id] = {str(doc.metadata["id"]): int(doc.metadata["id"] in rel_docs) 
                            for doc in res["documents"]}
        run["time_seconds"][q_id] = res["time_seconds"]

    return run


def save_run(run: Run, save_dir: str) -> None:
    with open(f'{save_dir}.json', 'w') as f:
        json.dump(run, f)

    print(f"Run {save_dir} saved.")


def evaluate_retrieval(file: str, 
                       qrels_dict: Mapping[str, Mapping[str, int]], 
                       metrics: Union[list[str], str], 
                       return_mean: bool = True) -> tuple[dict[str, float], pd.DataFrame]:
    
    
    qrels = ranx.Qrels(qrels_dict)
    
    with open(file, 'r') as f:
        results = json.load(f)

    run = ranx.Run(results["results"])

    eval_results = ranx.evaluate(qrels, run, metrics, return_mean=return_mean, return_std=True)
  
    df = pd.DataFrame(data=eval_results.values(), index=eval_results.keys())

    return eval_results, df


def compare_evaluations(files: list[str], 
                        run_names: list[str],
                        qrels_dict: Mapping[str, Mapping[str, int]], 
                        metrics: Union[list[str], str], 
                        stat_test: str, 
                        max_p: float):
    qrels = ranx.Qrels(qrels_dict)

    runs_list = []

    for file, run_name in zip(files, run_names):
        with open(file, 'r') as f:
            results = json.load(f)
        
        run = ranx.Run(results["results"])
        run.name = run_name

        runs_list.append(run)
    
    report = ranx.compare(
        qrels,
        runs=runs_list,
        metrics=metrics,
        max_p=max_p,  # P-value threshold
        stat_test=stat_test,
    )

    return report


def format_metrics(metrics: Union[list[str], str]) -> list[str]:
    if isinstance(metrics, str):
        metrics = [metrics]
    return metrics


def compare_evaluations_stats(files: list[str], 
                            run_names: list[str],
                            qrels_dict: Mapping[str, Mapping[str, int]], 
                            metrics: Union[list[str], str], 
                            stat_test: str, 
                            n_permutations: int = 1000,
                            max_p: float = 0.01,
                            random_seed: int = 42,
                            threads: int = 0,
                            rounding_digits: int = 3,
                            show_percentages: bool = False,
                            make_comparable: bool = False,
                            correction: Optional[str] = None):
    
    assert stat_test == "fisher", "This functionality currently only supports Fisher Randomization."
    assert correction in [None, "holm-bonf"], "This functionality currently only supports Holm-Bonferroni correction."
    
    qrels = ranx.Qrels(qrels_dict)

    runs_list = []
    time_seconds = []

    for file, run_name in zip(files, run_names):
        with open(file, 'r') as f:
            results = json.load(f)
        
        run = ranx.Run(results["results"])
        run.name = run_name

        runs_list.append(run)

        time_seconds.append(results["time_seconds"])
    
    metrics = format_metrics(metrics)
    assert all(isinstance(m, str) for m in metrics), "Metrics error"

    model_names = []
    results = defaultdict(dict)


    metric_scores = {}

 
    for i, run in enumerate(runs_list):
        model_name = run.name if run.name is not None else f"run_{i+1}"
        model_names.append(model_name)

        metric_scores[model_name] = ranx.evaluate(
            qrels=qrels,
            run=run,
            metrics=metrics,
            return_mean=False,
            threads=threads,
            make_comparable=make_comparable,
        )

        metric_scores[model_name]["time_seconds"] = np.array(list(time_seconds[i].values()))

        if len(metrics) == 1:
            metric_scores[model_name] = {metrics[0]: metric_scores[model_name]}

        for m in metrics:
            results[model_name][m] = float(np.mean(metric_scores[model_name][m]))
        
        results[model_name]["time_seconds"] = float(np.mean(
                                                        list(time_seconds[i].values())
                                                    ))

    comparisons = ranx.statistical_tests.compute_statistical_significance(
        model_names=model_names,
        metric_scores=metric_scores,
        stat_test=stat_test,
        n_permutations=n_permutations,
        max_p=max_p,
        random_seed=random_seed,
    )

    if correction:
        print(f"Correcting p_values using {correction}...")
        comparisons = correct_p(comparisons=comparisons, 
                                correction=correction,
                                alpha=max_p)
    
    all_stats, all_pairwise_stats = extract_stats(model_names=model_names, 
                                                  metrics=metrics, 
                                                  comparisons=comparisons, 
                                                  metric_scores=metric_scores)

    
    win_tie_loss = defaultdict(dict)

    metrics_with_time = metrics + ["time_seconds"]

    for control in model_names:
        for treatment in model_names:
            if control != treatment:
                for m in metrics_with_time:
                    control_scores = metric_scores[control][m]
                    treatment_scores = metric_scores[treatment][m]
                    win_tie_loss[(control, treatment)][m] = {
                        "W": int(sum(control_scores > treatment_scores)),
                        "T": int(sum(control_scores == treatment_scores)),
                        "L": int(sum(control_scores < treatment_scores)),
                    }
    

    # custom report with inference time metric
    report = Report(
        model_names=model_names,
        results=dict(results),
        comparisons=comparisons,
        metrics=metrics_with_time,
        max_p=max_p,
        win_tie_loss=dict(win_tie_loss),
        rounding_digits=rounding_digits,
        show_percentages=show_percentages,
        stat_test=stat_test,
    )

    return report, all_stats, all_pairwise_stats 



def correct_p(comparisons: FrozensetDict, 
            correction: Optional[str] = None, 
            alpha: float = 0.05) -> FrozensetDict:
    
    all_p_values = []
    corr_comparisons = FrozensetDict()
    
    if correction == "holm-bonf":
        for pair, metrics_dict in comparisons.items():
            for metric, stats_dict in metrics_dict.items():
                all_p_values.append(stats_dict["p_value"])

        corrected_results = multipletests(all_p_values, alpha=alpha, method='holm')
        corrected_p_values = corrected_results[1]
        corrected_significant = corrected_results[0]

        corr_pvals_iter = iter(corrected_p_values.tolist())
        corr_sig_iter = iter(corrected_significant.tolist())

        for pair, metrics_dict in comparisons.items():
            control, treatment = pair
            corr_vals = {}
            for metric, stats_dict in metrics_dict.items():
                corr_vals[metric] = {
                    "p_value": next(corr_pvals_iter),
                    "significant": next(corr_sig_iter)
                }

            corr_comparisons[frozenset([control, treatment])] = corr_vals
        
    else:
        raise NotImplementedError(f"Correction method `{correction}` not supported.")
        

    return corr_comparisons


def bootstrap_ci_and_se(scores, n_bootstrap=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ]
    mean_diff = float(np.mean(boot_means))
    se = float(np.std(boot_means))
    lower = (100 - ci) / 2
    upper = 100 - lower
    ci_bounds = np.percentile(boot_means, [lower, upper])

    return {
        "mean-diff": mean_diff, 
        "std-error": se, 
        "confidence-interval": ci_bounds.tolist()
    }


def cliffs_delta(a, b) -> float:
    n = len(a)
    m = len(b)
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    return float((greater - less) / (n * m))


def interpret_cliffs_delta(delta: float) -> str:
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "N"
    elif abs_delta < 0.33:
        return "S"
    elif abs_delta < 0.474:
        return "M"
    else:
        return "L"


def extract_stats(model_names, metrics, metric_scores, comparisons: FrozensetDict):
    all_stats = {} 
    all_pairwise_stats = {}

    for model_name in model_names:
        all_stats[model_name] = {}
        for m in metrics + ["time_seconds"]:
            scores = metric_scores[model_name][m]
            std = float(np.std(scores))
            ci_lower, ci_upper, se = bootstrap_ci_and_se(scores)
            all_stats[model_name][m] = {
                "scores": scores,
                "mean": float(np.mean(scores)), 
                "std": std,
                "std-error": float(std / np.sqrt(len(scores))), 
                "std-error-bootstrap": se, 
                "confidence-interval": [ci_lower, ci_upper]
            } 
  

    for pair, metrics_dict in comparisons.items():
        control, treatment = pair
        all_pairwise_stats[(control, treatment)] = {}
        for m, stats_dict in metrics_dict.items():
            control_scores = all_stats[control][m]["scores"]
            treatment_scores = all_stats[treatment][m]["scores"]
            diff = treatment_scores - control_scores
            bootstrap_stats = bootstrap_ci_and_se(diff)
            delta = cliffs_delta(treatment_scores, control_scores)

           
            all_pairwise_stats[(control, treatment)][m] = {
                "mean-diff": bootstrap_stats["mean-diff"], 
                "std-error": bootstrap_stats["std-error"], 
                "confidence-interval": bootstrap_stats["confidence-interval"], 
                "cliffs-delta": delta
            }
        

    return all_stats, all_pairwise_stats


def plot_forest(all_pairwise_stats, base_model: str, metrics_directions: dict[str, bool]):
    metrics = metrics_directions.keys()
    rows = []

    for (control, treatment), stats in all_pairwise_stats.items():
        if base_model not in (control, treatment):
            continue

        for metric in metrics:
            if metric not in stats:
                continue

            stat = stats[metric]
            mean_diff = stat.get('mean-diff')
            ci_low, ci_high = stat.get('confidence-interval')
            delta = stat.get("cliffs-delta")

            if any(x is None for x in (mean_diff, ci_low, ci_high, delta)):
                continue

            if control == base_model:
                other = treatment
                mean_diff = -mean_diff
                ci_low, ci_high = -ci_high, -ci_low
                delta = -delta
            else:
                other = control

            if not metrics_directions.get(metric, True):
                mean_diff = -mean_diff
                ci_low, ci_high = -ci_low, -ci_high
                delta = -delta

            rows.append({
                "Comparison": f"{other}",
                "Metric": metric,
                "Mean": mean_diff,
                "CI Low": ci_low,
                "CI High": ci_high,
                "Cliff's Delta": delta
            })

    if not rows:
        print(f"[Warning] No comparisons found involving base model '{base_model}'.")
        return

    df = pd.DataFrame(rows)
    df["Comparison"] = pd.Categorical(df["Comparison"], categories=sorted(df["Comparison"].unique()), ordered=True)
    offset = [-0.2, 0.0, 0.2, 0.4, 0.6]
    df["Metric Offset"] = df["Metric"].map({
        metric: offset[i] for i, metric in enumerate(metrics)
    })

    df["Plot Y"] = df["Comparison"].cat.codes + df["Metric Offset"]

    plt.figure(figsize=(12, len(df["Comparison"].unique()) * 1))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("deep", n_colors=len(metrics))
    metric_color = dict(zip(metrics, palette))
    seen_labels = set()

    for _, row in df.iterrows():
        label = row["Metric"] if row["Metric"] not in seen_labels else None
        ci_low, ci_high = sorted([row["CI Low"], row["CI High"]])
        xerr = [[row["Mean"] - ci_low], [ci_high - row["Mean"]]]

        plt.errorbar(
            x=row["Mean"],
            y=row["Plot Y"],
            xerr=xerr,
            fmt='o',
            color=metric_color[row["Metric"]],
            capsize=4,
            label=label
        )
        seen_labels.add(row["Metric"])

        effect_size = interpret_cliffs_delta(row["Cliff's Delta"])
        padding = 0.03 if len(metrics) < 2 else 0.002
        text_x = max(row["Mean"], row["CI High"]) + padding
        plt.text(
            text_x,
            row["Plot Y"],
            f"δ = {row['Cliff\'s Delta']:.2f} ({effect_size})",
            fontsize=9,
            va='center',
            ha='left'
        )

    # Axes
    plt.axvline(x=0, color='gray', linestyle='dashed')
    plt.yticks(
        ticks=range(len(df["Comparison"].cat.categories)),
        labels=df["Comparison"].cat.categories
    )
    plt.xlabel(f"\nMean Difference\nPositive favors {base_model}")
    plt.title(f"Forest Plot: {base_model} vs. Other Methods\n")

    # Metric legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Cliff's Delta effect size legend with thresholds
    effect_legend = [
        Line2D([0], [0], marker='', color='none', label="Effect sizes (|δ|):"),
        Line2D([0], [0], marker='', color='none', label="N = negligible (|δ| < 0.147)"),
        Line2D([0], [0], marker='', color='none', label="S = small (0.147 ≤ |δ| < 0.33)"),
        Line2D([0], [0], marker='', color='none', label="M = medium (0.33 ≤ |δ| < 0.474)"),
        Line2D([0], [0], marker='', color='none', label="L = large (≥ 0.474)"),
    ]

    plt.legend(
        list(by_label.values()) + effect_legend,
        list(by_label.keys()) + [e.get_label() for e in effect_legend],
        title="Metric / Effect Size",
        loc='lower right'
    )

    plt.tight_layout()
    plt.show()



def flatten_properties(data: dict[str, Any], prefix: str = "", separator: str = "_") -> dict[str, Any]:
    """
    Recursively flatten nested dictionaries into flat key-value pairs with only primitive values.
    
    Args:
        data: Dictionary to flatten
        prefix: Prefix to add to keys
        separator: Character to use between nested keys
    
    Returns:
        Flattened dictionary with only primitive values
    """
    flattened = {}
    
    if not isinstance(data, dict):
        return data
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_properties(value, new_key, separator))
        elif isinstance(value, list):
            if any(isinstance(item, dict) for item in value):
                flattened[new_key] = json.dumps(value)
            else:
                flattened[new_key] = value
        else:
            flattened[new_key] = value
    
    return flattened


def rebuild_graph_documents(graph_documents):
    """
    Completely rebuild graph documents from scratch to eliminate any nested object issues
    """

    rebuilt_documents = []
    
    for doc_idx, doc in enumerate(graph_documents):
        print(f"Processing document {doc_idx}...")
        
        rebuilt_nodes = []
        node_map = {}  # To track nodes for relationship building
        
        for node_idx, node in enumerate(doc.nodes):
            try:
                # Extract only primitive properties
                clean_props = {}
                if hasattr(node, 'properties') and node.properties:
                    for key, value in node.properties.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_props[key] = value
                        elif isinstance(value, dict):
                            # Flatten nested dicts
                            flattened = flatten_properties(value, key)
                            for flat_key, flat_value in flattened.items():
                                if isinstance(flat_value, (str, int, float, bool)):
                                    clean_props[flat_key] = flat_value
                        elif isinstance(value, list):
                            # Only keep lists of primitives
                            if all(isinstance(item, (str, int, float, bool)) for item in value):
                                clean_props[key] = value
                
                # Create completely new node
                new_node = Node(
                    id=str(node.id),
                    type=str(node.type), 
                    properties=clean_props
                )
                
                rebuilt_nodes.append(new_node)
                node_map[node.id] = new_node
                
            except Exception as e:
                print(f"Error processing node {node_idx}: {e}")
                continue
        
        rebuilt_relationships = []
        
        for rel_idx, rel in enumerate(doc.relationships):
            try:
                # Extract only primitive properties for relationships
                clean_props = {}
                if hasattr(rel, 'properties') and rel.properties:
                    for key, value in rel.properties.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_props[key] = value
                        elif isinstance(value, dict):
                            # Flatten nested dicts
                            flattened = flatten_properties(value, key)
                            for flat_key, flat_value in flattened.items():
                                if isinstance(flat_value, (str, int, float, bool)):
                                    clean_props[flat_key] = flat_value
                
                # Find source and target nodes in our rebuilt set
                source_node = node_map.get(rel.source.id)
                target_node = node_map.get(rel.target.id)
                
                if source_node and target_node:
                    new_relationship = Relationship(
                        source=source_node,
                        target=target_node,
                        type=str(rel.type),
                        properties=clean_props
                    )
                    rebuilt_relationships.append(new_relationship)
                else:
                    print(f"Warning: Could not find source/target for relationship {rel_idx}")
                    
            except Exception as e:
                print(f"Error processing relationship {rel_idx}: {e}")
                continue
        
        # Create new document
        try:
            new_doc = GraphDocument(
                nodes=rebuilt_nodes,
                relationships=rebuilt_relationships,
                source=doc.source if hasattr(doc, 'source') else None
            )
            rebuilt_documents.append(new_doc)
        except Exception as e:
            print(f"Error creating document {doc_idx}: {e}")
            continue
    
    return rebuilt_documents



import numpy as np
import pandas as pd
from siamics.data.geo import GEO
from siamics.data import drop_sparse_data
import GEOparse, os, logging, pickle, json
from scipy.stats import spearmanr, kruskal
from scipy.linalg import svd as scipy_svd
import statsmodels.api as sm
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings(
    "ignore",
    message="Downcasting behavior in `replace` is deprecated*",
    category=FutureWarning
)

logging.getLogger("GEOparse").setLevel(logging.WARNING)

# def get_rem_gse(catalogue, stats, nb_genes=19062, threshold=0.5, min_samples=15):
#     filtered_catalogue = drop_sparse_data(catalogue, stats, nb_genes, threshold)
#     gsm_counts = filtered_catalogue["group_id"].value_counts()
#     valid_gse_ids = gsm_counts[gsm_counts >= min_samples].index
#     return valid_gse_ids.to_numpy()

def replace_custom_nas(series, na_values=None):
    if na_values is None:
        na_values = {
            "", "nan", "NaN", "NAN", "null", "NULL", "None", "NONE",
            "NA", "N/A", "n/a", "na", "--", "-", "/", "\\", "not applicable", "none"
        }
    return series.replace(list(na_values), np.nan).infer_objects(copy=False)

def parse_soft(soft_dir, gse_id, gsm_ids, output_dir, logging=False):
    filepath = os.path.join(soft_dir, f"{gse_id}_family.soft.gz")
    gse = GEOparse.get_GEO(filepath=filepath, how="full")

    rows = []

    for gsm_id, gsm in gse.gsms.items():
        if gsm_id not in gsm_ids:
            continue

        row = {"gsm_id": gsm_id}

        # for characteristics_ch1 - multiple fields
        for key, value in gsm.metadata.items():
            if key == "characteristics_ch1":
                for entry in value:
                    if ":" in entry:
                        k, v = entry.split(":", 1)
                        row[k.strip().lower()] = v.strip()
            else:
                row[key] = "; ".join(value) if isinstance(value, list) else value
        
        rows.append(row)

    metadata_df = pd.DataFrame(rows)

    # drop columns: if study has sup files, content with links (e.g. points to SRA files)
    cols_to_drop = [col for col in metadata_df.columns if col.startswith("supplementary_file")]
    for col in metadata_df.columns:
        if metadata_df[col].astype(str).str.contains(r'(?:http|https|ftp)://', regex=True).any():
            cols_to_drop.append(col)

    # remove col with only one unique val - can't check biological sig
    # metadata_df = metadata_df.loc[:, metadata_df.nunique(dropna=False) > 1] 
    nunique = metadata_df.nunique(dropna=False)
    metadata_df = metadata_df.loc[:, (nunique > 1)]

    metadata_df = metadata_df.drop(columns=[col for col in set(cols_to_drop) if col in metadata_df.columns])

    transposed = metadata_df.T
    deduplicated = transposed.drop_duplicates() # drop factors with the exact same values
    metadata_df = deduplicated.T

    # Ensure gsm_id is retained
    if "gsm_id" not in metadata_df.columns:
        metadata_df["gsm_id"] = [row["gsm_id"] for row in rows]

    id = ["id", "identifier"]
    id_escaped = [re.escape(f) for f in id]
    id_pattern = r"(?<![a-zA-Z0-9])(" + "|".join(id_escaped) + r")(?![a-zA-Z0-9])|_(" + "|".join(id_escaped) + r")(?![a-zA-Z0-9])"
    compiled_id_pattern = re.compile(id_pattern)

    id_related_cols = []
    for col in metadata_df.columns:
        col_lower = col.lower()

        if col_lower == "gsm_id": #ensure gsm_id is retrained
            continue

        series = metadata_df[col]

        # Drop 'title' only if all values are unique
        if col_lower == "title" and series.nunique() == len(series):
            id_related_cols.append(col)

        # Drop 'id' only if all values are unique
        elif compiled_id_pattern.search(col_lower) and series.nunique() == len(series):
            id_related_cols.append(col)

    if logging:
        output_o_meta = os.path.join(output_dir, "metadata_full.csv")
        metadata_df.to_csv(output_o_meta, index=False)

    metadata_df = metadata_df.drop(columns=id_related_cols)
    metadata_df = metadata_df.apply(lambda col: replace_custom_nas(col) if col.dtype == "object" else col)

    if logging:
        output_path = os.path.join(output_dir, "metadata.csv")
        metadata_df.to_csv(output_path, index=False)

    return metadata_df

# assign cat/cont to a variable
def infer_column_status(col_series):
    col = col_series.dropna().astype(str).str.strip()

    numeric_values = pd.to_numeric(col, errors="coerce")
    num_total = len(col)

    # If 90% value fails, not numeric enough
    if numeric_values.notna().sum() < num_total * 0.9: 
        status = "categorical"
    elif (numeric_values % 1 != 0).any():
        status = "continuous"
    else:
        status = "continuous" if numeric_values.nunique(dropna=False) == len(col_series) else "categorical"

    # all_unique = col_series.nunique(dropna=False) == len(col_series)

    # return status, all_unique
    return status

# generating num mapping for str categorical vars
def map_categorical_columns(metadata_df, status_df, output_dir, logging=False):
    category_mappings = {}

    for col, status in status_df["status"].items():
        if status != "categorical":
            continue

        original_series = metadata_df[col]
        series = original_series.astype(str).str.strip()

        # Treat known placeholders in GEO as NaN
        series = replace_custom_nas(series)
        # series = series.replace({
        #     "": np.nan, "nan": np.nan, "NaN": np.nan, "null": np.nan,
        #     "None": np.nan, "NA": np.nan, "N/A": np.nan,
        #     "-": np.nan, "/": np.nan, "n/a": np.nan, "na": np.nan, "--": np.nan,
        #     "not applicable": np.nan, "none": np.nan, "null": np.nan
        # })

        non_na_series = series.dropna()
        numeric_check = pd.to_numeric(non_na_series, errors="coerce")

        if numeric_check.notna().all():
            continue

        # Map only non-numeric categories
        non_numeric_cats = non_na_series[numeric_check.isna()].unique()

        if len(non_numeric_cats) > 0:
            used_values = set(numeric_check[numeric_check.notna()].astype(float))

            num = ["age", "bmi", "time", "day", "days", "duration", "hour", "hours", "hrs", "weight", "week", "weeks", "height", "length", "months", "year", "years", "number"]

            factor_num = [re.escape(factor) for factor in num]
            pattern = r"(?<![a-zA-Z0-9])(" + "|".join(factor_num) + r")(?![a-zA-Z0-9])|_(" + "|".join(factor_num) + r")(?![a-zA-Z0-9])"
            compiled = re.compile(pattern)

            matches = bool(compiled.search(col))

            def custom_parse(value):
                if not isinstance(value, str):
                    value = str(value)

                digits = re.findall(r"\d+(?:\.\d+)?", value)
                if not digits:
                    return None
                
                if "-" in value or "_" in value:
                    if len(digits) >= 2:
                        try:
                            nums = [float(d) for d in digits[:2]]
                            return sum(nums) / 2
                        except:
                            return None
                    elif len(digits) == 1:
                        return float(digits[0])

                if any(sep in value for sep in [",", ".", "+"]): # + for week+day (e.g. gestation), "," for bmi
                    if len(digits) >= 2:
                        try:
                            return float(f"{digits[0]}.{digits[1]}")
                        except:
                            return None
                    elif len(digits) == 1:
                        return float(digits[0])

                try:
                    return float(digits[0])
                except:
                    return None

            if matches:
                values_extracted = [custom_parse(item) for item in non_numeric_cats]
                used_values.update(val for val in values_extracted if val is not None)

                mapping = {}
                fallback_val = 0

                for cat, val in zip(non_numeric_cats, values_extracted):
                    if val is not None:
                        mapping[cat] = val
                    else:
                        # Find next unused float for fallback
                        while fallback_val in used_values:
                            fallback_val += 1
                        mapping[cat] = float(fallback_val)
                        used_values.add(fallback_val)
                        fallback_val += 1

                if any(val is not None and val % 1 != 0 for val in values_extracted):
                    status_df.loc[col, "status"] = "continuous"
            else:
                mapping = {}
                fallback_val = 0.0
                for cat in sorted(non_numeric_cats):
                    # Avoid assigning a fallback value already used in numeric part
                    while fallback_val in used_values:
                        fallback_val += 1
                    mapping[cat] = fallback_val
                    used_values.add(fallback_val)
                    fallback_val += 1

            category_mappings[col] = mapping

            mapped_series = series.map(mapping)
            metadata_df[col] = mapped_series.astype("float")

    if logging:
        output_meta_path = os.path.join(output_dir, "metadata_mapped.csv")
        metadata_df.to_csv(output_meta_path, index=False)
        output_mapping_path = os.path.join(output_dir, "mapping.csv")

    if logging:
        flat_mapping = [
            {"column": col, "category": k, "code": v}
            for col, mapping in category_mappings.items()
            for k, v in mapping.items()
        ]
        pd.DataFrame(flat_mapping).to_csv(output_mapping_path, index=False)

    return metadata_df, category_mappings, status_df 

def load_and_concat_expression(gsm_pkl_list_path, target_gse, catalogue):
    with open(gsm_pkl_list_path, "rb") as f:
        all_files = pickle.load(f)

    gse_files = [
        f for f in all_files
        if f"{target_gse}_norm_counts" in os.path.dirname(f)
    ]

    # print(f"Found {len(gse_files)} files for {target_gse}")

    valid_sample_ids = set(catalogue["sample_id"].astype(str).str.strip())

    # Keep only files whose name (without .pkl) matches a sample_id
    matched_files = [
        f for f in gse_files
        if os.path.splitext(os.path.basename(f))[0] in valid_sample_ids
    ]

    # print(f"{len(matched_files)} files matched with catalogue sample_ids")

    expr_list = [pd.read_pickle(f) for f in matched_files]
    expr = pd.concat(expr_list).T  # gene x sample

    return expr

def filter_expression_matrix(expr, min_non_nan_fraction=0.75, sample_var_cutoff=0.01):
    min_non_nan_count = int(min_non_nan_fraction * expr.shape[1])
    filtered_genes_expr = expr[expr.notna().sum(axis=1) >= min_non_nan_count]

    sample_variance = filtered_genes_expr.var(axis=0)
    filtered_expr = filtered_genes_expr.loc[:, sample_variance >= sample_var_cutoff]

    return filtered_expr

def impute_missing_with_gene_means(expr, verbose=False):
    nan_mask = expr.isna()

    if nan_mask.values.any():
        gene_means = expr.mean(axis=1)
        imputed_expr = expr.T.fillna(gene_means).T
        if verbose:
            print("Missing values imputed with gene means.")
            print(f"Missing value count: {nan_mask.sum().sum()}")
    else:
        if verbose:
            print("There are no missing values.")
        imputed_expr = expr

    return imputed_expr

def row_standardize(matrix, eps=1e-6):
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    stds = stds.mask(stds < eps, eps)
    return matrix.sub(means, axis=0).div(stds, axis=0)

# iteratively standardize rows and cols; threshold taken from PavlidisLab/baseCode/MatrixStats.java
def double_standardize(matrix, max_iters=100, tol_mean_row=0.02, tol_mean_col=0.1, tol_std=0.1, verbose=False):
    X = matrix.copy()
    converged = False

    for i in range(max_iters):
        X = row_standardize(X.T).T  # standardize columns
        X = row_standardize(X)      # standardize rows

        # Check convergence
        row_means = X.mean(axis=1)
        col_means = X.mean(axis=0)
        row_stds = X.std(axis=1)
        col_stds = X.std(axis=0)

        row_mean_ok = (row_means.abs() <= tol_mean_row).all()
        col_mean_ok = (col_means.abs() <= tol_mean_col).all()
        row_std_ok  = ((row_stds - 1).abs() <= tol_std).all()
        col_std_ok  = ((col_stds - 1).abs() <= tol_std).all()

        # if verbose:
        #     print(f"[Iteration {i+1}] mean(row) max: {row_means.abs().max():.4f}, mean(col) max: {col_means.abs().max():.4f}, std(row) max dev: {(row_stds - 1).abs().max():.4f}, std(col) max dev: {(col_stds - 1).abs().max():.4f}")

        if row_mean_ok and col_mean_ok and row_std_ok and col_std_ok:
            converged = True
            if verbose:
                print("Converged.")
            break
    else:
        if verbose:
            print("Max iterations reached. May not be fully converged.")

    return X, converged

def merge_pcs_with_metadata(VT, metadata_df, status_df, sample_ids, top_k=5):
    # PCs = VT[:top_k, :].T
    available_pcs = min(top_k, VT.shape[0])
    PCs = VT[:available_pcs, :].T
    PC_df = pd.DataFrame(PCs, columns=[f"PC{i+1}" for i in range(available_pcs)])
    PC_df["gsm_id"] = sample_ids

    # Merge PCs with metadata
    merged_df = PC_df.merge(metadata_df, on="gsm_id")

    factor_cols = [col for col in metadata_df.columns if col != "gsm_id"]
    factor_types = status_df["status"].to_dict()

    return merged_df, factor_cols, factor_types

def identify_important_factors(merged_df, factor_cols, factor_types, p_val_cutoff=0.01, verbose=False):
    important_factors_set = set()
    results = []

    pc_cols = [col for col in merged_df.columns if col.startswith("PC")]

    batch_escaped = [re.escape("batch")]
    pattern = (
        r"(?<![a-zA-Z0-9])(" + "|".join(batch_escaped) +
        r")(?![a-zA-Z0-9])|_(" + "|".join(batch_escaped) + r")(?![a-zA-Z0-9])"
    )
    batch_pattern = re.compile(pattern, flags=re.IGNORECASE)

    tech_factors_df = pd.read_csv("/projects/ovcare/users/tina_zhang/outlier/batch_filtered.csv")
    tech_factors = tech_factors_df["column_name"]

    for pc in pc_cols:
        for factor in factor_cols:
            if factor in important_factors_set:
                continue

            if batch_pattern.search(factor):
                if verbose:
                    print(f"skipping batch factor: {factor}")
                continue

            if factor in set(tech_factors.astype(str)):
                if verbose:
                    print(f"skipping technical factor: {factor}")
                continue

            df = merged_df[[pc, factor]].dropna()
            if len(df) < 3:
                if verbose:
                    print(f"Not enough data to analyze factor {factor}; skipping.")
                continue

            x = df[pc]
            y = df[factor]
            ftype = factor_types.get(factor, "categorical")
            pval = None

            if ftype == "continuous":
                _, pval = spearmanr(x, y)
            else:
                n_groups = y.nunique()
                if n_groups < 2:
                    if verbose:
                        print(f"All samples belong to the same group in factor {factor}; skipping.")
                    continue
                elif n_groups == len(y):
                    if verbose:
                        print(f"All samples belong to different categories in factor {factor}; skipping.")
                    continue
                elif n_groups == 2:
                    _, pval = spearmanr(x, y)
                else:
                    try:
                        groups = [x[y == g] for g in y.unique()]
                        _, p_kw = kruskal(*groups)
                        _, p_sp = spearmanr(x, y)
                        pval = min(p_kw, p_sp)
                    except Exception as e:
                        print(f"Error analyzing factor {factor}: {e}")
                        continue

            if pval is not None and pval < p_val_cutoff:
                results.append({"PC": pc, "factor": factor, "p_value": pval})
                important_factors_set.add(factor)

    if verbose:
        print("Biologically important factors (excluding batch):")
        print(important_factors_set)

    return important_factors_set, results

def regress_out_factors(filtered_expr, metadata_df, filtered_factors, min_non_missing_frac=0.7, verbose=False):
    filtered_factors_sorted = sorted(filtered_factors)
    metadata = metadata_df.set_index("gsm_id")
    
    design = metadata.loc[filtered_expr.columns, filtered_factors_sorted]
    design = design.apply(pd.to_numeric, errors='coerce')

    dropped_factors = []

    # Drop factors with too many NaNs
    while True:
        factor_nan_fractions = design.isna().mean(axis=0)
        to_drop = factor_nan_fractions[factor_nan_fractions > (1 - min_non_missing_frac)].index.tolist()
        if not to_drop:
            break
        if verbose:
            print(f"Dropping factors with too much missing data: {to_drop}")
        dropped_factors.extend(to_drop)
        design = design.drop(columns=to_drop)
        if design.shape[1] == 0:
            reason = "All factors dropped due to missingness"
            if verbose:
                print(reason)
            return None, True, dropped_factors, reason

    # Drop samples with missing data
    num_samples_with_nan = design.isna().any(axis=1).sum()
    total_samples = design.shape[0]
    missing_frac = num_samples_with_nan / total_samples
    if verbose:
        print(f"Samples with missing factor values: {num_samples_with_nan}/{total_samples}")

    if missing_frac > (1 - min_non_missing_frac):
        reason = "Too many samples with missing values in remaining factors"
        if verbose:
            print(f"{reason}. Skipping regression.")
        return None, True, dropped_factors, reason

    design = design.dropna()
    if design.shape[0] < int(min_non_missing_frac * total_samples):
        reason = "Not enough samples after dropping rows with missing values"
        if verbose:
            print(f"{reason}. Skipping regression.")
        return None, True, dropped_factors, reason

    design = sm.add_constant(design) # intercept

    expr = filtered_expr.loc[:, design.index]

    X = design.values
    Y = expr.values
    Y_T = Y.T

    B = np.linalg.lstsq(X, Y_T, rcond=None)[0]
    Y_pred_T = X @ B
    Y_pred = Y_pred_T.T
    residuals_matrix = Y - Y_pred

    # Add back gene-wise means
    gene_means = expr.mean(axis=1).to_numpy().reshape(-1, 1)
    residuals_matrix += gene_means

    regressed_matrix = pd.DataFrame(residuals_matrix, index=expr.index, columns=expr.columns)
    return regressed_matrix, False, dropped_factors, None

def find_desired_quantile_index(n, quantile_threshold):
    p = quantile_threshold / 100.0
    if p < (2.0 / 3.0) / (n + (1.0 / 3.0)): # avoid negative indices
        return 1.0
    elif p >= (n - (1.0 / 3.0)) / (n + (1.0 / 3.0)): # avoid accessing non-existing values
        return float(n)
    else:
        return ((n + 1.0 / 3.0) * p) + (1.0 / 3.0)

def find_value_at_desired_quantile(arr, quantile_threshold):
    arr = np.sort(arr[~np.isnan(arr)])
    n = len(arr)
    if n == 0:
        return np.nan

    h = find_desired_quantile_index(n, quantile_threshold)
    h_floor = int(np.floor(h))
    h_frac = h - h_floor

    lower = arr[h_floor - 1] 
    upper = arr[h_floor] if h_floor < n else arr[-1]

    return lower + h_frac * (upper - lower)

def removeFalsePositives(all_samples, outliers, q1_dict, q3_dict):
    inliers = [s for s in all_samples if s not in outliers]

    if not inliers:
        return outliers

    inliers_sorted = sorted(inliers, key=lambda s: q1_dict[s])
    threshold = q1_dict[inliers_sorted[0]]

    while True:
        for i, sample in enumerate(outliers):
            if q3_dict[sample] >= threshold:
                # revise threshold and remove sample from outlier list
                threshold = q1_dict[sample]
                outliers = outliers[:i] + outliers[i+1:]
                break
        else:
            break

    return outliers

def run_geo_outlier_pipeline(catalogue, root_dir, gsm_list_path, logging=True, verbose=True):
    # get GSEs with remaining samples after dropping sparse:
    # full_catalogue = pd.read_csv("/projects/ovcare/users/tina_zhang/data/GEO/catalogue.csv")
    # dataset = GEO(catalogue=full_catalogue)
    # catalogue = dataset.catalogue

    # stats = pd.read_csv("/projects/ovcare/users/tina_zhang/data/GEO/stats_train.csv") 
    # gse_list = get_rem_gse(catalogue, stats) 

    gse_list = catalogue['group_id'].unique().tolist()
    not_converged_gse = []
    skipped_gse = []
    soft_dir = os.path.join(root_dir, "softs")
    all_outlier_pairs = []
    refined_outliers_path = os.path.join(root_dir, "outliers", "all_refined_outliers.csv")

    report_path = os.path.join(root_dir, "outliers", "all_gse_reports.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if not os.path.exists(report_path):
        with open(report_path, "w") as f:
            f.write("[\n")

    first_entry = True

    for gse_id in tqdm(gse_list, desc="Processing GSEs"):
        gsm_ids = catalogue.loc[catalogue['group_id'] == gse_id, 'sample_id'].tolist()

        output_dir = os.path.join(root_dir, "outliers", gse_id)

        if logging:
            os.makedirs(output_dir, exist_ok=True)

        print(f"------Analyzing {gse_id}-------")

        print("Parsing SOFT file")
        metadata = parse_soft(soft_dir, gse_id, gsm_ids, output_dir, logging)
        factor_status = {
            col: infer_column_status(metadata[col])
            for col in metadata.columns if col != "gsm_id" 
        }

        print("Preparing factors")
        status = pd.DataFrame.from_dict(factor_status, orient="index", columns=["status"])
        
        metadata_mapped, _, status = map_categorical_columns(metadata, status, output_dir, logging)

        if logging:
            output_status_path = os.path.join(output_dir, "var_status.csv")
            status.to_csv(output_status_path)
        
        print("Preparing expression matrices")
        expr = load_and_concat_expression(gsm_list_path, gse_id, catalogue)
        filtered_expr = filter_expression_matrix(expr)
        if filtered_expr.shape[0] < filtered_expr.shape[1]:
            reason = "More samples than genes after filtering"
            if verbose:
                print(f"SVD not supported: {reason}. Skipping {gse_id}.")
            skipped_gse.append((gse_id, reason))
            continue
        elif filtered_expr.shape[0] == 0:
            reason = "No genes remaining after filtering"
            if verbose:
                print(f"SVD not supported: {reason}. Skipping {gse_id}.")
            skipped_gse.append((gse_id, reason))
            continue
        else:
            filtered_expr = impute_missing_with_gene_means(expr=filtered_expr, verbose=verbose)

        # double standarization
        norm_expr, converged = double_standardize(filtered_expr)
        if not converged:
            not_converged_gse.append(gse_id)

        print("SVD")
        expr = norm_expr.values
        if expr.shape[1] < 2:
            reason = f"Only {expr.shape[1]} sample(s) available"
            if verbose:
                print(f"SVD not supported: {reason}. Skipping {gse_id}.")
            skipped_gse.append((gse_id, reason))
            continue

        if np.isnan(expr).any() or np.isinf(expr).any():
            reason = "NaN or Inf in matrix after standardization"
            if verbose:
                print(f"SVD aborted: {reason}. Skipping {gse_id}.")
            skipped_gse.append((gse_id, reason))
            continue

        # conventions following Gemma's comments:
        #  * Perform SVD on an expression data matrix, E = U S V'. The rows of the input matrix are probes (genes), following the
        #  * convention of Alter et al. 2000 (PNAS). Thus the U matrix columns are the <em>eigensamples</em> (eigenarrays) and the
        #  * V matrix columns are the <em>eigengenes</em>. See also http://genome-www.stanford.edu/SVD/.
        try:
            U, S, VT = np.linalg.svd(expr, full_matrices=False) #rows of VT = eigenvectors; Sigma already sorted in descending order

        except np.linalg.LinAlgError:
            print(f"np.linalg.svd failed for {gse_id}, trying scipy.linalg.svd...")
            try:
                U, S, VT = scipy_svd(expr, full_matrices=False)
                print("scipy.linalg.svd succeeded.")
            except Exception as e:
                reason = f"SVD did not converge even with fallback: {e}"
                print(f"Skipping {gse_id}: {reason}")
                skipped_gse.append((gse_id, reason))
                continue

        print("Identifying biologically important factors")
        merged_df, factor_cols, factor_types = merge_pcs_with_metadata(VT, metadata_mapped, status, sample_ids=filtered_expr.columns)

        filtered_factors, _ = identify_important_factors(merged_df=merged_df, factor_cols=factor_cols, factor_types=factor_types, verbose=verbose)

        if not filtered_factors:
            reason = "No biologically important factors identified"
            if verbose:
                print(f"{reason}. Skipping {gse_id}.")
            skipped_gse.append((gse_id, reason))
            continue

        print("Regressing biologically important factors")
        regressed_matrix, skipped, dropped_factors, skip_reason = regress_out_factors(
            filtered_expr=filtered_expr, metadata_df=metadata_mapped, filtered_factors=filtered_factors, verbose=verbose
        )

        gse_report = {
            "gse_id": gse_id,
            "important_factors": list(filtered_factors),
            "dropped_factors_due_to_missingness": dropped_factors,
            "regression_skipped": skipped,
            "skip_reason": skip_reason
        }

        with open(report_path, "a") as f:
            if not first_entry:
                f.write(",\n")
            json.dump(gse_report, f)
            first_entry = False

        if skipped:
            skipped_gse.append((gse_id, f"regression skipped: {skip_reason}"))
            continue

        print("Outlier Detection")
        corr_matrix = np.corrcoef(regressed_matrix.to_numpy().T) 
        np.fill_diagonal(corr_matrix, np.nan) # mask self-correlations

        # compute row corr stats
        q1, q2, q3 = [], [], []

        for row in corr_matrix:
            q1.append(find_value_at_desired_quantile(row, 25))
            q2.append(find_value_at_desired_quantile(row, 50))
            q3.append(find_value_at_desired_quantile(row, 75))

        sample_ids = regressed_matrix.columns

        stats_df = pd.DataFrame({
            "sample_id": sample_ids,
            "Q1": q1,
            "median": q2,
            "Q3": q3
        })

        stats_df = stats_df.sort_values("median").reset_index(drop=True)

        outlier_cut_index = None
        for i in range(len(stats_df) - 1):
            if stats_df.loc[i, "Q3"] < stats_df.loc[i + 1, "Q1"]:
                outlier_cut_index = i
                break

        if outlier_cut_index is not None:
            outlier_samples = stats_df.loc[:outlier_cut_index, "sample_id"].tolist()
        else:
            outlier_samples = []

        # print(f"Detected {len(outlier_samples)} outliers.")

        if len(outlier_samples) != 0:
            all_samples = stats_df["sample_id"].tolist()
            q1_dict = stats_df.set_index("sample_id")["Q1"].to_dict()
            q3_dict = stats_df.set_index("sample_id")["Q3"].to_dict()
            refined_outliers = removeFalsePositives(all_samples, outlier_samples, q1_dict, q3_dict)

            if refined_outliers:
                print(f"Refined outliers detected in {gse_id}: {len(refined_outliers)}")
                outlier_df = pd.DataFrame({
                    "gse_id": [gse_id] * len(refined_outliers),
                    "sample_id": refined_outliers
                })

                write_header = not os.path.exists(refined_outliers_path) or os.path.getsize(refined_outliers_path) == 0

                outlier_df.to_csv(refined_outliers_path, mode="a", header=write_header, index=False)
                all_outlier_pairs.extend([(gse_id, sid) for sid in refined_outliers])

    with open(report_path, "a") as f:
        f.write("\n]\n")

    if skipped_gse:
        skipped_df = pd.DataFrame(skipped_gse, columns=["gse_id", "reason"])
        skipped_path = os.path.join(root_dir, "outliers", "skipped_gse.csv")
        skipped_df.to_csv(skipped_path, index=False)

    if not_converged_gse:
        not_converged_df = pd.DataFrame(not_converged_gse, columns=["gse_id"])
        not_converged_path = os.path.join(root_dir, "outliers", "not_converged_gse.csv")
        not_converged_df.to_csv(not_converged_path, index=False)

    return all_outlier_pairs
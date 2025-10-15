from pathlib import Path
from typing import List, Tuple
import xarray as xr, numpy as np, pandas as pd
import tqdm.auto as tqdm, functools
import typing

def get_best_shift(diffs: np.ndarray, tol: float):
    if diffs.size==0:
        raise Exception("No data")
    diff_min = diffs - tol
    diff_max = diffs + tol
    merged = np.concatenate((diff_min, diff_max))
    order = merged.argsort()
    sorted_merged = merged[order]
    counts = np.cumsum(2*(order<diff_min.size) - 1)
    best_match = counts.argmax()
    n_matches = counts[best_match]
    min_shift = sorted_merged[best_match]
    max_shift = sorted_merged[best_match+1]
    if np.isnan(min_shift+max_shift):
        raise Exception("GOT NAN")
    return n_matches, min_shift, max_shift
      

class SlopeSearch(typing.NamedTuple):
    min: float
    max: float
    n_branches: int
    precision: float

    @property
    def n_iters(self) -> int:
        if self.max - self.min < self.precision:
            return 1
        return self.n_branches*int(np.floor(np.log2((self.max-self.min)/self.precision)/np.log2(self.n_branches))+1)
    
    def optimize(self, func):
        ncalls=0
        test_values = []
        res_values = []
        curr = (self.max + self.min)/2
        precision = self.max - curr
        if precision < self.precision:
            curr_val = func(curr)[1]
            ncalls+=1
        while precision > self.precision:
            test_space = np.linspace(curr-precision, curr + precision, self.n_branches, endpoint=False) + precision/self.n_branches
            vals = [func(x) for x in test_space]
            test_values+=list(test_space)
            res_values+=[v[0] for v in vals]
            best = np.argmax([v[0] for v in vals])
            curr_val = vals[best][1]
            curr = test_space[best]
            precision = precision/self.n_branches
            ncalls+=self.n_branches
        if ncalls!=self.n_iters:
            print(f"n_iter error, got {ncalls} calls, expected {self.n_iters}")
        return curr, curr_val
    

class TolSearch(typing.NamedTuple):
    min: float
    max: float
    precision: float
    percentage: float =0.9

    @property
    def n_iters(self) -> int:
        return 1+int(np.floor(np.log2((self.max-self.min)/self.precision))+1)
    
    def optimize(self, func):
        min_tol = self.min
        max_tol = self.max
        max_val, state = func(max_tol)
        min_val = 0
        while max_tol - min_tol > self.precision:
            curr = (max_tol + min_tol)/2
            val, s = func(curr)
            if val > self.percentage*(max_val - min_val):
                max_tol=curr
                max_val, state = val,s
            else:
                min_val = val
                min_tol=curr
        return (max_tol + min_tol)/2, state




def compute_initial_sync_values(ref_arrs: List[np.ndarray], rel_arrs: List[np.ndarray], 
                                tol_search: TolSearch = TolSearch(0, 0.050, 10**-3, 0.95), 
                                slope_search: SlopeSearch = SlopeSearch(0.999, 1.001, 20, 10**-6), 

                                progress: tqdm.tqdm | None= None
                                ) -> Tuple[int, float, float, float]:
    tol_search = TolSearch(*tol_search)
    slope_search = SlopeSearch(*slope_search)
    if len(ref_arrs) != len(rel_arrs):
       raise Exception("Expected same number of categories")
    if progress is None:
       progress = tqdm.tqdm(desc="Computing initial sync values", total=tol_search.n_iters*slope_search.n_iters)
    with progress:
        computed_diff_arrs = {}
        def get_diff_arr(slope: float):
            rng = np.random.default_rng()
            if not slope in computed_diff_arrs:
                marr = [(ref_arrs[group][~np.isnan(ref_arrs[group])], rel_arrs[group][~np.isnan(rel_arrs[group])]) for group in range(len(ref_arrs))]
                total_size = sum([ref.size*rel.size for ref, rel in marr])
                if total_size > 3*10**5:
                    factor = np.sqrt(total_size / (3*10**5))
                    marr = [(rng.choice(ref, int(ref.size/factor)), rng.choice(rel, int(rel.size/factor))) for ref, rel in marr]
                res= np.concatenate([(ref.reshape(-1, 1) - slope*rel.reshape(1, -1)).flatten() for ref, rel in marr])
                computed_diff_arrs[slope] =res
            return computed_diff_arrs[slope]

        def evaluate_from_slope_and_shift(tol, slope):
            progress.update()
            progress.set_postfix(tol=tol, slope=slope)
            diffs = get_diff_arr(slope)
            n, min_shift, max_shift = get_best_shift(diffs, tol)
            return n, (n, min_shift, max_shift)

        def evaluate_from_tol(tol):
            slope, (n, min_shift, max_shift) = slope_search.optimize(lambda slope: evaluate_from_slope_and_shift(tol, slope))
            return n, (n, slope, min_shift, max_shift)

        tol, (n, slope, min_shift, max_shift) = tol_search.optimize(evaluate_from_tol)
    return n, tol, slope, min_shift, max_shift



def compute_match(ref_ar, rel_ar, tol, cost_func, progress):
    initial_cost = (0, 0)
    cache = {}
    stack = [(0, 0, False)] 

    while stack:
        i, j, expanded = stack.pop()

        if (i, j) in cache:
            continue

        if expanded is not False:
            progress.update()
            results = []
            for (ci, cj), (incnb, incdiff, added) in expanded:
                (nb, diff), matching = cache[(ci, cj)]
                result = ((nb+incnb, diff+incdiff), added+matching)
                results.append(result)
            try:
                cache[(i, j)] = min(results, key=lambda r: cost_func(*r[0]))
            except:
                print(i, j, len(ref_ar), len(rel_ar))
                raise
        else:
            cases = []
            if i < ref_ar.size and j < rel_ar.size:
                ta = ref_ar[i]
                tb = rel_ar[j]
                diff = abs(ta - tb)
                if diff < tol:
                    cases.append(((i + 1, j + 1), (1, diff, [(i, j)])))
                if tb - ta < tol:
                    cases.append(((i, j + 1), (0, 0, [])))
                if ta - tb < tol:
                    cases.append(((i + 1, j), (0, 0, [])))
                stack.append((i, j, cases))
                for (ci, _) in cases:
                    stack.append((ci[0], ci[1], False))
            else:
                cache[(i, j)] = initial_cost, []
            
            
    return cache[(0, 0)]

def compute_match_intervals(ref_ar_start, rel_ar_start, ref_ar_end, rel_ar_end, tol, 
            cost_func = lambda n, diff: diff-n, 
            interval_func =lambda n_ref, n_rel, ref_hole_dur, rel_hole_dur, ref_dur, rel_dur: (ref_hole_dur+rel_hole_dur)/(ref_dur+rel_dur) < 0.5,
            progress: tqdm.tqdm | None = None
):
    if progress is None:
        progress = tqdm.tqdm(desc="computing_match")
    with progress:
        _, grp_start = compute_match(ref_ar_start, rel_ar_start, tol, cost_func, progress)
        _, grp_end = compute_match(ref_ar_end, rel_ar_end, tol, cost_func, progress)
    def find_start_end_matches(grp_starts, grp_ends):
        s, e = 0, 0
        groups = []
        ref_group_id_start = np.full(ref_ar_start.size, -1)
        ref_group_id_end = np.full(ref_ar_start.size, -1)
        rel_group_id_start = np.full(rel_ar_start.size, -1)
        rel_group_id_end = np.full(rel_ar_start.size, -1)

        for s in range(len(grp_starts)):
            match = False
            if e < len(grp_ends):
                try:
                    if (s+1 >= len(grp_starts)) or  (grp_ends[e][0] < grp_starts[s+1][0]):
                        def compute_holes(starts, ends, match_s, match_e):
                            holes_start = starts[match_s+1:match_e+1]
                            holes_end = ends[match_s:match_e]
                            return match_e - match_s, np.sum(holes_start - holes_end), ends[e] - starts[s]
                        n_ref_holes, ref_hole_duration, ref_duration = compute_holes(ref_ar_start, ref_ar_end, grp_starts[s][0], grp_ends[e][0])
                        n_rel_holes, rel_hole_duration, rel_duration = compute_holes(rel_ar_start, rel_ar_end, grp_starts[s][1], grp_ends[e][1])
                        if interval_func(n_ref_holes, n_rel_holes, ref_hole_duration, rel_hole_duration, ref_duration, rel_duration):
                            match = True
                except:
                    print(grp_starts)
                    print(grp_starts[s+1])
                    raise
            if match:
                for i in range(grp_starts[s][0], grp_ends[e][0]+1):
                    ref_group_id_start[i] = len(groups)
                    ref_group_id_end[i] = len(groups)
                for i in range(grp_starts[s][1], grp_ends[e][1]+1):
                    rel_group_id_start[i] = len(groups)
                    rel_group_id_end[i] = len(groups)
                groups.append((grp_starts[s][0], grp_ends[e][0], grp_starts[s][1], grp_ends[e][1], 
                               n_ref_holes, ref_hole_duration, n_rel_holes, rel_hole_duration, 
                               ))
                e+=1
            else:
                ref_group_id_start[grp_starts[s][0]] = len(groups)
                rel_group_id_start[grp_starts[s][1]] = len(groups)
                groups.append((grp_starts[s][0], -1, grp_starts[s][1], -1, pd.NA, pd.NA, pd.NA, pd.NA))
            while e < len(grp_ends) and ((s+1 >= len(grp_starts)) or grp_ends[e][0] < grp_starts[s+1][0]):
                ref_group_id_end[grp_ends[e][0]] = len(groups)
                rel_group_id_end[grp_ends[e][1]] = len(groups)
                groups.append((-1, grp_ends[e][0], -1, grp_ends[e][1], pd.NA, pd.NA, pd.NA, pd.NA))
                e+=1
        while e < len(grp_ends):
            ref_group_id_end[grp_ends[e][0]] = len(groups)
            rel_group_id_end[grp_ends[e][1]] = len(groups)
            groups.append((-1, grp_ends[e][0], -1, grp_ends[e][1], pd.NA, pd.NA, pd.NA, pd.NA))
            e+=1
        return groups, np.stack([ref_group_id_start, ref_group_id_end], axis=-1), np.stack([rel_group_id_start, rel_group_id_end], axis=-1)
    res, ref_ev, rel_ev = find_start_end_matches(grp_start, grp_end)
    ds = xr.Dataset()
    if len(res) > 0:
        ds["ev_ind"] = xr.DataArray([[[v[0], v[1]], [v[2], v[3]]] for v in res], dims=["match_group", "which", "bound"])
        ds["n_holes"] = xr.DataArray([[v[4], v[6]] for v in res], dims=["match_group", "which"])
        ds["hole_duration"] = xr.DataArray([[v[5], v[7]] for v in res], dims=["match_group", "which"])
    else:
        ds["ev_ind"] = xr.DataArray(np.full((0, 2, 2), 0, dtype=np.int64), dims=["match_group", "which", "bound"])
        ds["n_holes"] = xr.DataArray(np.full((0, 2), 0, dtype=np.int64), dims=["match_group", "which"])
        ds["hole_duration"] = xr.DataArray(np.full((0, 2), np.nan, dtype=np.float64), dims=["match_group", "which"])
    ds["which"] = ["ref", "rel"]
    ds["bound"] = ["start", "end"]
    ds["match_group"] = np.arange(len(res))

    return ds, xr.DataArray(ref_ev, dims=["event", "bound"]).to_dataset(name="match_group"), xr.DataArray(rel_ev, dims=["event", "bound"]).to_dataset(name="match_group")

    

import numpy as np, pandas as pd, xarray as xr
from typing import Literal, Annotated
import functools
import tqdm.auto as tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import xarray as xr, numpy as np, pandas as pd
import shutil
import plotly.colors as pc



def get_log_colorscale(source_colorscale="plasma", lowest=6, npoints=20):
    return pc.sample_colorscale(pc.get_colorscale(source_colorscale), 1-np.geomspace(1, 10**-lowest, npoints))
def get_plotly_config(filename):
  return  {'scrollZoom': True, 'displaylogo': False, 'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp,
        'filename': filename,
    }}
def facet_distplot(ds: xr.Dataset, x: str, y: str, mode: Literal["scatter", "heatmap"]="scatter", 
                   facet_row: str | None =None, facet_col: str | None=None, color: str| None=None, 
                   facet_row_agg: Literal["color", "pattern_shape"]  ="pattern_shape",
                   facet_col_agg: Literal["color", "pattern_shape"] | None = "pattern_shape",
                   nbinsx: int=None, nbinsy: int=None, hoverdata=None):
    if facet_col is None:
       facet_col = "_facet_col"
       ds[facet_col]=xr.DataArray([0], dims=facet_col)
       
    if facet_row is None:
       facet_row = "_facet_row"
       ds[facet_row]=xr.DataArray([0], dims=facet_row)
       ds[facet_row] = np.array([0])

    if set(ds[facet_row].dims).intersection(set(ds[facet_col].dims)):
        raise Exception("Row and column facetting need to be distinct dimensions")
    row_grouped = list(ds.groupby(facet_row))
    col_grouped = list(ds.groupby(facet_col))
    fig = make_subplots(len(row_grouped)+1, len(col_grouped)+1,
                         shared_xaxes="all", shared_yaxes=True, 
                         column_titles=[v[0] for v in col_grouped]+["counts"], row_titles=["counts"]+[v[0] for v in row_grouped], vertical_spacing=0.02,
                         row_heights=[1]+ [2]*len(row_grouped), column_widths=[2]*len(col_grouped)+[1])
    rows = {}
    cols={}
    for i, (r, g) in enumerate(row_grouped):
        agg_col =  facet_col if len(col_grouped) > 1 else None
        agg_color = color
        agg_pattern_shape = None
        if agg_col and facet_row_agg=="color":
           agg_color = agg_col 
        if agg_col and facet_row_agg=="pattern_shape":
           agg_pattern_shape=agg_col

        hist = px.histogram(g.to_dataframe().reset_index(), y=y, nbins=nbinsy, color=agg_color, pattern_shape = agg_pattern_shape, barmode="stack")
        hist.update_traces(showlegend=i==0 and facet_col != "_facet_col")
        # if has_both_facets:
        if facet_col != "_facet_col":
          hist.update_traces(legendgroup="columns", legendgrouptitle={'text': facet_col})
        for trace in hist.select_traces():
            fig.add_trace(trace, col=len(col_grouped)+1, row=i+2)
        rows[r] = i
    for i, (r, g) in enumerate(col_grouped):
        agg_col =  facet_row if len(row_grouped) > 1 else None
        agg_color = color
        agg_pattern_shape = None
        if agg_col and facet_col_agg=="color":
           agg_color = agg_col 
        if agg_col and facet_col_agg=="pattern_shape":
           agg_pattern_shape=agg_col

        hist = px.histogram(g.to_dataframe().reset_index(), x=x, nbins=nbinsx, color=agg_color, pattern_shape = agg_pattern_shape, barmode="stack")
        
        hist.update_traces(showlegend=i==0 and facet_row != "_facet_row")
        if facet_row != "_facet_row":
          hist.update_traces(legendgroup="rows", legendgrouptitle={'text': facet_row})
        for trace in hist.select_traces():
            fig.add_trace(trace, col=i+1, row=1)
        cols[r] = i
    for i, (r, gtmp) in enumerate(col_grouped):
        for j, (row, g) in  enumerate(gtmp.groupby(facet_row)):
          if mode=="scatter":
            data = px.scatter(g.to_dataframe().reset_index(),x=x, y=y, color=color, trendline="lowess", hover_data=hoverdata)
          elif mode=="heatmap":
            data = px.density_heatmap(g.to_dataframe().reset_index(),x=x, y=y, nbinsx=nbinsx, nbinsy=nbinsy)
          row_index = rows[row]
          col_index= i
          data.update_traces(showlegend=i==0 and j==0, legendgroup="data", legendgrouptitle={'text': "data"})
          data.update_traces(showlegend=False, selector=dict(mode="lines"))
          for trace in data.select_traces():
            fig.add_trace(trace, row=row_index+2, col=col_index+1)
    
    fig.update_layout(barmode='stack', coloraxis_colorbar=dict(orientation="h"))
    fig.update_xaxes(title=x, row=len(row_grouped)+1)
    fig.update_xaxes(title=f"{y} count", col=len(col_grouped)+1, row=len(row_grouped)+1)
    tmp = {k:fig.layout[k]["title"]["text"] for k in fig.layout if "xaxis" in k}
    tmp = [k for k, v in tmp.items() if v==f"{y} count"][0].replace("xaxis", "x")
    fig.update_xaxes(matches=tmp, col=len(col_grouped)+1)

    fig.update_yaxes(title=y, col=1)
    fig.update_yaxes(title=f"{x} count", row=1, col=1)
    tmp = {k:fig.layout[k]["title"]["text"] for k in fig.layout if "yaxis" in k}
    tmp = [k for k, v in tmp.items() if v==f"{x} count"][0].replace("yaxis", "y")
    fig.update_yaxes(matches=tmp, selector=dict(title=dict(text=f"{x} count")))
    
    return fig


def plot_type_info(stats):
    # events_matching_metadata = events_matching_metadata[["match_group"]].to_dataframe().reset_index()
    # events_matching_metadata["match"] = events_matching_metadata["match_group"] > 0
    # missing =  px.histogram(events_matching_metadata,
    #          x="event_name", pattern_shape="bound", color="which", 
    #          barmode="group", text_auto=True)

    # data = matching_info.assign(
    #   n_ref_events_in_grp = (matching_info["ev_index"].sel(bound="end", which="ref") - matching_info["ev_index"].sel(bound="start", which="ref")+1).fillna(0).astype(int),
    #   n_rel_events_in_grp = (matching_info["ev_index"].sel(bound="end", which="rel") - matching_info["ev_index"].sel(bound="start", which="rel")+1).fillna(0).astype(int)
    # )[["n_ref_events_in_grp", "n_rel_events_in_grp", "event_name"]].to_dataframe().reset_index()
    # merges = px.density_heatmap(data, x="n_ref_events_in_grp", y="n_rel_events_in_grp", facet_col="event_name", text_auto=True, color_continuous_scale=get_log_colorscale())
    # merges.update_layout(coloraxis_showscale=False)
    # merges.update_xaxes(type='category')
    # merges.update_yaxes(type='category')
    fig = px.bar(
        stats.to_array(dim="type", name="count").to_dataframe().reset_index(),
        x="type",
        y="count",
        color="which",
        barmode="group",
        pattern_shape="bound",
        facet_col="event_name",
        text_auto=True,
    )
    return fig

def plot_time_info(matching_info, slope, shift):
    data: xr.Dataset= matching_info.copy()
    data = data.assign(
      ref_t = matching_info["t"].sel(which="ref") ,
      diff = matching_info["t"].sel(which="ref")- (matching_info["t"].sel(which="rel")*slope + shift)
    )
    fig = facet_distplot(data.drop_dims("which"), facet_col="event_name", x="ref_t", y="diff", color="bound", nbinsx=50, nbinsy=20, hoverdata=["match_group"])
    fig.update_traces(marker={'size': 2}, selector=dict(mode= 'markers'))
    return fig

def plot_matching(slope, shift, matching_info: xr.Dataset, event_ds, event_list):
    event_ds = event_ds.copy()
    event_ds["t"] = xr.where(event_ds["which"] == "rel", event_ds["t"]*slope+shift, event_ds["t"])
    event_ds["start"] = event_ds["t"].sel(bound="start")
    event_ds["end"] = event_ds["t"].sel(bound="end")
    event_ds["match_group"] = xr.where(event_ds["match_group"]<0, np.nan, event_ds["match_group"])
    
    matching_info = matching_info.copy()
    
    matching_info["t"] = xr.where(matching_info["which"] == "rel", matching_info["t"]*slope+shift, matching_info["t"])
    fig = make_subplots(rows=len(event_list), cols=1, shared_xaxes=True, shared_yaxes=True, row_titles=event_list, 
                        vertical_spacing=0.02)
    event_to_row = {k:i+1 for i,k in enumerate(event_list)}
    which_to_y = dict(ref=[0.8, 1.2], rel=[-0.2, 0.2])
    which_to_color = dict(ref="blue", rel="red")
    
    for i in range(event_ds.sizes["event"]):
        ev = event_ds.isel(event=i)
        fig.add_trace(go.Scatter(
            x=[ev["t"].sel(bound="start").item(), ev["t"].sel(bound="start").item(), 
               ev["t"].sel(bound="end").item(), ev["t"].sel(bound="end").item()], 
            y=[which_to_y[ev["which"].item()][0], which_to_y[ev["which"].item()][1], 
               which_to_y[ev["which"].item()][1], which_to_y[ev["which"].item()][0]], 
            # showlegend=not ev["which"].item() in has_arr, 
            showlegend=False,
            name="<br>".join(
                [k+"="+str(ev[k].item()) for k in ["event_id"]]+
                ["start="+str(np.round(ev["t"].sel(bound="start").item(), 4))] +
                ["end="+str(np.round(ev["t"].sel(bound="end").item(), 4))] + 
                ["duration="+str(np.round(ev["t"].sel(bound="end") - ev["t"].sel(bound="start"), 4).item())]
            ),
            # name=ev["which"].item() + "_events",
            fill="toself",
            fillcolor=which_to_color[ev["which"].item()],
            # opacity=0.5,
            mode="lines",
            line=dict(width=0)
        ), row=event_to_row[ev["event_name"].item()], col=1)
        
    for n in "ref", "rel":
       fig.add_trace(go.Scatter(x=[0,0,0, 0], y=[0, 0, 0, 0], showlegend=True,
            name=n+" events", fill="toself", fillcolor=which_to_color[n], opacity=0.5,
            mode="lines", line=dict(width=0)), row=1, col=1)
    
    missing_slices = {}
    event_ds = event_ds.drop_vars("event")
    for w in ["ref", "rel"]:
        ar = matching_info.sel(which=w)
        for ev, g in ar.groupby("event_name"):
            missing_slices[(ev, w)] = []
            missing_start_index = (g["ev_index"].sel(bound="end").shift(match_group=1, fill_value=-1)).to_numpy()
            missing_end_index = g["ev_index"].sel(bound="start").to_numpy()
            for s, e in zip(missing_start_index, missing_end_index):
                if (e - s >1) and s>=0 and e>=0:
                    missing_slices[(ev, w)].append((s+1, e-1))
    n_slices=0
    for (ev, w), g in event_ds.groupby(["event_name", "which"]):
       for (s, e) in missing_slices.get((ev, w), []):
            data = dict(start=g["t"].sel(bound="start").where(g["ev_index"] == s, drop=True).item(),
                        end=g["t"].sel(bound="end").where(g["ev_index"] == e, drop=True).item(),
                        n_missed=e-s+1
            )
            fig.add_trace(go.Scatter(
                x=[data["start"], data["end"]],
                y=[np.mean(which_to_y[w]), np.mean(which_to_y[w])],
                hovertext="<br>".join([f"{k}=" + str(data[k]) for k in ["n_missed", "start", "end"]]),
                line_color="black",
                line_width=1,
                mode="lines",
                showlegend=n_slices==0,
                name="missing_slices",
            ), row=event_to_row[ev], col=1)
            n_slices+=1


    matching_info["n_match_total"] = (matching_info["n_holes"] + 1).sum("which")
    diff_t = np.abs(matching_info["t"].sel(which="ref") - matching_info["t"].sel(which="rel"))
    matching_info["diff_t"] = diff_t.mean("bound")
    diff_t_max = matching_info["diff_t"].max()
    matching_info["t"] = matching_info["t"].fillna(matching_info["t"].sel(bound="start"))
    for i in range(matching_info.sizes["match_group"]):
        g = matching_info.isel(match_group=i)
        n = "primary matches" if g["n_match_total"].item() == 2 else "other matches"
        fig.add_trace(go.Scatter(
            x=[g["t"].sel(bound="start", which="ref").item(), g["t"].sel(bound="start", which="rel").item(), 
               g["t"].sel(bound="end", which="rel").item(), g["t"].sel(bound="end", which="ref").item()], 
            y=[which_to_y["ref"][0], which_to_y["rel"][1], which_to_y["rel"][1], which_to_y["ref"][0]],  
            name="<br>".join([f"{k}=" + str(g[k].item()) for k in ["match_group", "diff_t", "n_match_total"]]),
            fill="toself",
            fillcolor=f"rgba(0, 255, 0, {0.5+0.5*float(g['diff_t']/diff_t_max)})",
            line_color="rgba(0, 0, 0, 1)",
            mode="lines",
            showlegend=False,
            line=dict(width=0 if g["n_match_total"].item() == 2 else 1)), row=event_to_row[g["event_name"].item()], col=1)
    
    for n in "primary", "other":
       fig.add_trace(go.Scatter(
            x=[0, 0, 0, 0], 
            y=[0, 0, 0, 0],  
            fill="toself",
            fillcolor=f"rgba(0, 255, 0, 0.3)",
            line_color="rgba(0, 0, 0, 1)",
            mode="lines",
            showlegend=True,
            name=n+" matches",
            line=dict(width=0 if n == "primary" else 1)), row=1, col=1)
    
    
    fig.update_yaxes(
            tickmode = 'array',
            tickvals = [np.mean(v) for v in which_to_y.values()],
            ticktext = list(which_to_y.keys())
    )
    return fig
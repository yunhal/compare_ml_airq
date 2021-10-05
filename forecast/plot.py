import os, os.path
import pandas as pd
import json
import datetime
import logging
import plotly
import plotly.express as px
import plotly.graph_objects as go


# dash options include 'dash', 'dot', and 'dashdot'
def plot_lines(datals, xtitle, ytitle, title, saveas='fig.png', xgrid=False, ygrid=True):
    fig = go.Figure()
    for df, items in datals:
        for col, name, color, width, mode, dash in items:
            x = df.index
            y = df.loc[:,col]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    name=name,
                    line=dict(color=color, width=width, dash=dash),
                    marker=dict(
                        color=color,
                        size=width,
                        line=dict(
                            color=color,
                            width=0,
                        )
                    ),
                )
            )
    fig.update_layout(
        #title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        width=600, height=150,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            showgrid=xgrid,
            linewidth=2,
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
        yaxis=dict(
            showline=True,
            showgrid=ygrid,
            linewidth=2,
            zerolinewidth=2,
            zerolinecolor='rgb(169, 169, 169)',
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
    )
    parent = os.path.dirname(saveas)
    if len(parent) > 0:
        os.makedirs(parent, exist_ok=True)
    fig.write_image(saveas)
    return fig


def plot_distribution(df, col, nbin, xtitle, ytitle, title, saveas):
    fig = px.histogram(df, x=col, nbins=nbin, histnorm='probability density')
    fig.update_layout(
        #title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        width=600, height=300,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,
            showgrid=False,
            linewidth=2,
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            linewidth=2,
            zerolinewidth=2,
            zerolinecolor='rgb(169, 169, 169)',
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
    )
    parent = os.path.dirname(saveas)
    if len(parent) > 0:
        os.makedirs(parent, exist_ok=True)
    fig.write_image(saveas)
    return fig



def plot_scatter(df, xcol, ycol, xtitle, ytitle, title, saveas):
    fig = px.scatter(
        df, x=xcol, y=ycol,
        labels={
            xcol: xtitle,
            ycol: ytitle,
        },
    )
    fig.update_layout(
        #title=title,
        width=400, height=400,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            showline=True,
            showgrid=False,
            linewidth=2,
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            linewidth=2,
            zerolinewidth=2,
            zerolinecolor='rgb(169, 169, 169)',
            linecolor='rgb(169, 169, 169)',
            gridcolor='rgb(232, 232, 232)',
        ),
    )
    min_d = df.min()
    max_d = df.max()
    min_v, max_v = min(min_d[xcol], min_d[ycol]), min(max_d[xcol], max_d[ycol])
    d_range = [0, int(max_v * 1.1)]
    fig.add_trace(
        go.Scatter(
            x=d_range,
            y=d_range,
            mode='lines',
            line=dict(color='rgb(96, 96, 96)', width=1, dash='dot')
        )
    )

    fig.update_xaxes(range=d_range)
    fig.update_yaxes(range=d_range)
    parent = os.path.dirname(saveas)
    if len(parent) > 0:
        os.makedirs(parent, exist_ok=True)
    fig.write_image(saveas)
    return fig



if __name__ == '__main__':
    pass

"""Build a readable summary report from measured experiment output files."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'; RES=ROOT/'results'
lines=['# HiLo-Fed Initial Research Prototype Report','',
       '> This report summarizes the latest recorded experiments on the supplied datasets. It reports measured outputs and does not claim HiLo-Fed currently outperforms all baselines.','']
lines += ['## Experiment configuration','']
for suffix,label in [('', 'METR-LA'),('_pems_bay','PEMS-BAY')]:
    meta_path=RES/f'run_metadata{suffix}.json'
    if not meta_path.exists(): continue
    meta=json.loads(meta_path.read_text())
    lines += [f'### {label}','',
      f"- Rows loaded: **{meta['rows_used']}** ({'fast chronological subset' if meta.get('fast_mode') else 'full supplied dataset'}; temporal window stride: {meta.get('window_stride', 'n/a')})",
      f"- Sensors: **{meta['sensor_count']}**",
      f"- Train / validation / test rows: **{meta['train_rows']} / {meta['validation_rows']} / {meta['test_rows']}**",
      f"- Clients / edges: **{meta['client_count']} / {meta['edge_count']}**",
      f"- Rounds: **{meta['rounds']}**; seed: **{meta['seed']}**; runtime: **{meta['runtime_seconds']:.2f}s**",
      f"- LoRA-style adapter parameters: **{meta['lora_adapter_trainable_parameters']} / {meta['full_model_trainable_parameters']}** ({meta['lora_trainable_parameter_ratio']*100:.1f}%)",
      f"- Parsed distance graph edges: **{meta.get('usable_distance_graph_edges', 0)}**; graph-neighbour history features enabled: **{meta.get('graph_neighbour_features_used', False)}**",'']
    metrics=pd.read_csv(RES/f'forecast_metrics{suffix}.csv')
    sixty=metrics[metrics.horizon_minutes==60][['method','rmse_mph','mae_mph']].copy()
    lines += ['## 60-minute forecasting results — '+label,'','| Method | RMSE (mph) | MAE (mph) |','|---|---:|---:|']
    for _,r in sixty.iterrows():
        lines.append(f"| {r.method} | {r.rmse_mph:.3f} | {r.mae_mph:.3f} |")
    costs=pd.read_csv(RES/f'communication_costs{suffix}.csv')
    finite=costs[pd.to_numeric(costs.total_bytes,errors='coerce').notna()].copy()
    fedavg=finite.loc[finite.method=='FedAvg','total_bytes']
    base=float(fedavg.iloc[0]) if len(fedavg) else float('nan')
    lines += ['', '## Communication accounting — '+label,'', '| Method | Client uplink bytes | Total measured bytes | Total bytes saved vs FedAvg |','|---|---:|---:|---:|']
    for _,r in finite.iterrows():
        saving=(1-float(r.total_bytes)/base)*100 if base and pd.notna(base) else float('nan')
        lines.append(f"| {r.method} | {int(r.client_upload_bytes)} | {int(r.total_bytes)} | {saving:.1f}% |")
    rounds=pd.read_csv(RES/f'round_logs{suffix}.csv')
    h=rounds[rounds.method=='HiLo-Fed']
    if len(h):
        lines += ['', '### HiLo-Fed communication control observations','',
          f"- Client uploads across rounds: **{int(h.client_uploads.sum())}**",
          f"- Forced uploads due to staleness cap: **{int(h.forced_uploads.sum())}**",
          f"- Cloud aggregation rounds (including final flush): **{int(h.cloud_aggregated.sum())}**",'']
lines += ['## Interpretation and measurement scope','',
'- The automated test suite validates the reconstructed implementation, not any missing original Antigravity source code.',
'- In these results, HiLo-Fed does not consistently achieve the lowest 60-minute forecasting error across both datasets. The system exposes an accuracy–communication trade-off; the results do not establish universal forecasting superiority.',
'- The primary tables use the full supplied files, 3 rounds, 6 clients and 3 edges, with temporal window stride 50. The separate multi-seed repeatability sweeps use 1,200-row fast subsets.',
'- Communication totals are measured from serialized update payloads and a 16-byte HiLo-Fed message header. They do not include network-stack overhead, energy, or measured wall-clock network latency.',
'- The forecaster consumes own-sensor history and distance-weighted graph-neighbour history. It is a graph-informed spatial-temporal linear model, not a deep message-passing GNN.',
'- Distance metadata is filtered to IDs present in the selected dataset, aligned to sensor order, normalized into a weighted adjacency matrix, and used to derive graph-neighbour input features.',
'- Centralized and non-trained reference baselines have blank communication totals because their transfer costs are not comparable to federated model-update traffic.',
'- The full-file experiments process every row in the supplied CSV files but sample every 50th eligible temporal window. They are multi-round full-file experiments, not exhaustive every-window runs.',
'- Multi-seed summary files are included for both datasets; their fast subsets provide repeatability evidence, not full-data confidence intervals.','']
report='\n'.join(lines)
(OUT/'INITIAL_RUN_REPORT.md').write_text(report,encoding='utf-8')
# Render a small standalone HTML report, including real tables.
import html

def render_html(src_lines):
    out=[]; i=0
    while i < len(src_lines):
        line=src_lines[i]
        if not line.strip():
            i+=1; continue
        if line.startswith('|'):
            block=[]
            while i < len(src_lines) and src_lines[i].startswith('|'):
                cells=[c.strip() for c in src_lines[i].strip().strip('|').split('|')]
                if not all(set(c) <= set('-: ') for c in cells): block.append(cells)
                i+=1
            if block:
                out.append('<table>')
                out.append('<thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in block[0])+'</tr></thead>')
                if len(block)>1:
                    out.append('<tbody>')
                    for row in block[1:]:
                        out.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>')
                    out.append('</tbody>')
                out.append('</table>')
            continue
        if line.startswith('# '): out.append('<h1>'+html.escape(line[2:])+'</h1>'); i+=1; continue
        if line.startswith('## '): out.append('<h2>'+html.escape(line[3:])+'</h2>'); i+=1; continue
        if line.startswith('### '): out.append('<h3>'+html.escape(line[4:])+'</h3>'); i+=1; continue
        if line.startswith('> '): out.append('<blockquote>'+html.escape(line[2:])+'</blockquote>'); i+=1; continue
        if line.startswith('- '):
            items=[]
            while i < len(src_lines) and src_lines[i].startswith('- '):
                items.append('<li>'+html.escape(src_lines[i][2:])+'</li>'); i+=1
            out.append('<ul>'+''.join(items)+'</ul>'); continue
        out.append('<p>'+html.escape(line)+'</p>'); i+=1
    return '\n'.join(out)

(OUT/'INITIAL_RUN_REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>HiLo-Fed Initial Run Report</title><style>body{font-family:Arial,sans-serif;max-width:1100px;margin:36px auto;line-height:1.5;padding:0 22px;color:#20252b}h1,h2,h3{margin-top:28px}table{border-collapse:collapse;width:100%;margin:12px 0 22px;font-size:14px}td,th{border:1px solid #c8cdd2;padding:7px 9px;text-align:left}th{background:#eef1f4}blockquote{border-left:4px solid #999;padding:8px 14px;margin-left:0;background:#f6f7f8}li{margin:5px 0}</style></head><body>'+render_html(lines)+'</body></html>',encoding='utf-8')
print('Wrote reports/INITIAL_RUN_REPORT.md and .html')

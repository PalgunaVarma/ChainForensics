import { useEffect, useState } from 'react';

const API = 'http://127.0.0.1:8000';
type Dataset = { id: string; source_name: string; unique_transactions: number; accepted_observations: number; flagged_count: number; rejected_count: number; duplicates: number; errors: { source_record: number; error: string }[] };
type Alert = { txid: string; flagged: boolean; anomaly_percentile: number; reasons: { feature: string; value: number; reference_median: number }[]; source_records: number[]; explanation_method: string };
type Graph = { nodes: { id: string; type: string; label: string }[]; edges: { id: string; source: string; target: string; amount_sats: number }[]; limitations: string };
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API + path, init);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
export default function BackendAnalysis() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selected, setSelected] = useState<Alert | null>(null);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [online, setOnline] = useState(false);
  useEffect(() => {
    let active = true;
    Promise.all([request('/health'), request<Dataset[]>('/datasets')]).then(([, rows]) => {
      if (active) { setOnline(true); setDatasets(rows); }
    }).catch(() => { if (active) setError('Backend unavailable. Start bash scripts/run-backend.sh in /home/doom/Documents.'); });
    return () => { active = false; };
  }, []);
  async function openDataset(row: Dataset) {
    const result = await request<{ alerts: Alert[] }>(`/datasets/${row.id}/alerts?flagged_only=false&limit=200`);
    setDataset(row); setAlerts(result.alerts); setSelected(null); setGraph(null);
  }
  async function action(work: () => Promise<void>) {
    setBusy(true); setError('');
    try { await work(); setOnline(true); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function accept(row: Dataset) {
    await openDataset(row);
    setDatasets(await request<Dataset[]>('/datasets'));
  }
  async function inspect(alert: Alert) {
    if (!dataset) return;
    const result = await request<Graph>(`/datasets/${dataset.id}/graph?txid=${alert.txid}`);
    setSelected(alert); setGraph(result);
  }
  async function exportDataset() {
    if (!dataset) return;
    const result = await request(`/datasets/${dataset.id}/export`);
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `chain-forensics-${dataset.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="panel content-panel backend-panel" aria-label="Backend analysis">
    <div className="eyebrow">CONNECTED ANALYSIS · {online ? 'API AVAILABLE' : 'API NOT YET VERIFIED'}</div>
    <h2>Analyze transaction evidence.</h2>
    <p>Real backend scoring with Isolation Forest against a synthetic reference. Anomaly percentiles are not probabilities of criminal activity. The other tabs retain their illustrative sample case.</p>
    <div style={{display:'flex', gap:12, flexWrap:'wrap', margin:'24px 0'}}>
      <button className="primary" disabled={busy} onClick={() => action(async () => accept(await request<Dataset>('/demo', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({count:100, seed:42})})))}>{busy ? 'Processing…' : 'Generate & analyze demo'}</button>
      <label style={{display:'grid', gap:8}}>Import JSON, CSV or XML (2 MiB max)
        <input type="file" accept=".json,.csv,.xml" disabled={busy} aria-label="Import dataset" onChange={e => {
          const file = e.target.files?.[0]; e.target.value = ''; if (!file) return;
          if (file.size > 2 * 1024 * 1024) {setError('Maximum file size is 2 MiB.'); return;}
          const form = new FormData(); form.append('file', file);
          action(async () => accept(await request<Dataset>('/datasets', {method:'POST', body:form})));
        }}/>
      </label>
      <select aria-label="Previous datasets" disabled={busy} value={dataset?.id || ''} onChange={e => { const row=datasets.find(d => d.id===e.target.value); if(row) action(() => openDataset(row)); }} style={{maxWidth:'100%',padding:10,background:'#302d26',color:'#eee',border:'1px solid #655b4b',borderRadius:5}}>
        <option value="">Choose a saved dataset</option>{datasets.map(d=><option key={d.id} value={d.id}>{d.source_name} · {d.id.slice(0,8)}</option>)}
      </select>
    </div>
    {error && <p role="alert" style={{color:'#ffc0a8',overflowWrap:'anywhere',marginBottom:20}}>{error}</p>}
    {dataset && <>
      <div className="stats" style={{marginBottom:20}}>{[['Transactions',dataset.unique_transactions],['Flagged',dataset.flagged_count],['Rejected records',dataset.rejected_count],['Duplicate records',dataset.duplicates]].map(([label,value])=><article className="stat" key={label}><div className="stat-heading">{label}</div><div className="stat-value">{value}</div></article>)}</div>
      <div style={{display:'flex',gap:16,alignItems:'center',flexWrap:'wrap',marginBottom:20}}><span>Dataset: {dataset.source_name}</span><button className="primary" disabled={busy} onClick={() => action(exportDataset)}>Export backend report</button></div>
      {dataset.errors.length>0 && <details><summary>Rejected record details</summary><ul>{dataset.errors.map(e=><li key={e.source_record}>Record {e.source_record}: {e.error}</li>)}</ul></details>}
      <div className="table-scroll"><table><thead><tr><th>TRANSACTION</th><th>ANOMALY PERCENTILE</th><th>STATUS</th><th>EVIDENCE</th></tr></thead><tbody>{alerts.map(a=><tr key={a.txid}><td className="mono" title={a.txid}>{a.txid.slice(0,18)}…</td><td>{a.anomaly_percentile.toFixed(1)}</td><td>{a.flagged ? 'Flagged for review' : 'Below demo threshold'}</td><td><button className="text-button" disabled={busy} onClick={() => action(() => inspect(a))} aria-label={`Inspect ${a.txid}`}>Inspect →</button></td></tr>)}</tbody></table></div>
      <p style={{marginTop:12}}>Showing {alerts.length} ranked transactions (maximum 200).</p>
    </>}
    {selected && <div style={{marginTop:28}}><h3>Transaction evidence</h3><p style={{overflowWrap:'anywhere',margin:'12px 0'}}>{selected.txid}</p><p>Source records: {selected.source_records.join(', ')}</p>
      <div className="table-scroll"><table><thead><tr><th>FEATURE</th><th>OBSERVED</th><th>REFERENCE MEDIAN</th></tr></thead><tbody>{selected.reasons.map(r=><tr key={r.feature}><td>{r.feature}</td><td>{r.value}</td><td>{r.reference_median}</td></tr>)}</tbody></table></div>
      <p style={{margin:'12px 0'}}>{selected.explanation_method}</p>
      {graph && <><h3>Address participation</h3><p>{graph.nodes.length} nodes · {graph.edges.length} edges. {graph.limitations}</p><div className="table-scroll"><table><thead><tr><th>FROM</th><th>TO</th><th>SATOSHIS</th></tr></thead><tbody>{graph.edges.map(e=><tr key={e.id}><td title={e.source}>{e.source.startsWith('tx:')?'This transaction':e.source.slice(8)}</td><td title={e.target}>{e.target.startsWith('tx:')?'This transaction':e.target.slice(8)}</td><td>{e.amount_sats.toLocaleString()}</td></tr>)}</tbody></table></div></>}
    </div>}
    {!dataset && <p>Generate a demo or select an existing dataset to start. Uploaded amounts must be integer satoshis.</p>}
  </section>;
}
